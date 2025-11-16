"""Resource management for CPU and GPU allocation."""

import os
import subprocess
import threading
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from pipeline_orchestrator.exceptions import ResourceError


class ResourceManager:
    """Manages CPU and GPU resource allocation."""
    
    def __init__(self, max_cpus: Optional[int] = None, max_gpus: Optional[int] = None):
        """
        Initialize resource manager.
        
        Args:
            max_cpus: Maximum number of CPUs to use (None for auto-detect)
            max_gpus: Maximum number of GPUs to use (None for auto-detect)
        """
        self.max_cpus = max_cpus or self._detect_cpu_count()
        self.max_gpus = max_gpus or self._detect_gpu_count()
        
        # Resource allocation tracking
        self._lock = threading.Lock()
        
        # Allocated to modules: {module_name: {"cpus": int, "gpus": List[int]}}
        self._allocated_to_modules: Dict[str, Dict] = {}
        
        # Borrowed for nested tasks: {task_id: {"cpus": int, "gpus": List[int]}}
        self._borrowed_for_nested: Dict[str, Dict] = {}
        
        # Available GPU devices (0-indexed)
        self._available_gpus = list(range(self.max_gpus))
        
        # Track which GPUs are in use: Set[int]
        self._gpu_in_use: Set[int] = set()
    
    def _detect_cpu_count(self) -> int:
        """Detect number of available CPUs."""
        try:
            import psutil
            return psutil.cpu_count(logical=True)
        except ImportError:
            # Fallback to os.cpu_count()
            count = os.cpu_count()
            return count if count else 1
    
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        # Try nvidia-smi first
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Try pynvml as fallback
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return count
        except (ImportError, Exception):
            pass
        
        # No GPUs detected
        return 0
    
    @property
    def total_cpus(self) -> int:
        """Total number of CPUs."""
        return self.max_cpus
    
    @property
    def total_gpus(self) -> int:
        """Total number of GPUs."""
        return self.max_gpus
    
    @property
    def allocated_cpus(self) -> int:
        """Number of CPUs currently allocated to modules."""
        with self._lock:
            return sum(alloc.get("cpus", 0) for alloc in self._allocated_to_modules.values())
    
    @property
    def allocated_gpus(self) -> int:
        """Number of GPUs currently allocated to modules."""
        with self._lock:
            return len(self._gpu_in_use)
    
    @property
    def borrowed_cpus(self) -> int:
        """Number of CPUs borrowed for nested tasks."""
        with self._lock:
            return sum(alloc.get("cpus", 0) for alloc in self._borrowed_for_nested.values())
    
    @property
    def borrowed_gpus(self) -> int:
        """Number of GPUs borrowed for nested tasks."""
        with self._lock:
            borrowed_gpu_set = set()
            for alloc in self._borrowed_for_nested.values():
                borrowed_gpu_set.update(alloc.get("gpus", []))
            return len(borrowed_gpu_set)
    
    @property
    def available_cpus(self) -> int:
        """Number of CPUs currently available."""
        return self.total_cpus - self.allocated_cpus - self.borrowed_cpus
    
    @property
    def available_gpus(self) -> int:
        """Number of GPUs currently available."""
        with self._lock:
            all_gpus = set()
            # GPUs in use by modules
            all_gpus.update(self._gpu_in_use)
            # GPUs borrowed for nested tasks
            for alloc in self._borrowed_for_nested.values():
                all_gpus.update(alloc.get("gpus", []))
            return self.total_gpus - len(all_gpus)
    
    def reserve_resources(
        self,
        module_name: str,
        cpus: int,
        gpus: int
    ) -> Tuple[List[int], Optional[str]]:
        """
        Reserve resources for a module.
        
        Args:
            module_name: Name of the module
            cpus: Number of CPUs required
            gpus: Number of GPUs required
            
        Returns:
            Tuple of (assigned_gpu_ids, cuda_visible_devices_string)
            
        Raises:
            ResourceError: If resources are not available
        """
        with self._lock:
            # Check if module already has resources allocated
            if module_name in self._allocated_to_modules:
                raise ResourceError(f"Module '{module_name}' already has resources allocated")
            
            # Validate resource availability
            if cpus > self.available_cpus:
                raise ResourceError(
                    f"Insufficient CPUs: requested {cpus}, available {self.available_cpus}"
                )
            
            if gpus > self.available_gpus:
                raise ResourceError(
                    f"Insufficient GPUs: requested {gpus}, available {self.available_gpus}"
                )
            
            # Allocate CPUs (just track count)
            # Allocate GPUs (assign specific devices)
            assigned_gpus = []
            if gpus > 0:
                available_gpu_list = [gpu for gpu in self._available_gpus if gpu not in self._gpu_in_use]
                if len(available_gpu_list) < gpus:
                    raise ResourceError(f"Insufficient GPU devices: requested {gpus}, available {len(available_gpu_list)}")
                
                assigned_gpus = available_gpu_list[:gpus]
                self._gpu_in_use.update(assigned_gpus)
            
            # Record allocation
            self._allocated_to_modules[module_name] = {
                "cpus": cpus,
                "gpus": assigned_gpus
            }
            
            # Create CUDA_VISIBLE_DEVICES string for process mode
            cuda_visible = None
            if assigned_gpus:
                # CUDA_VISIBLE_DEVICES uses comma-separated device indices
                # We map assigned GPU indices to sequential 0, 1, 2, ...
                cuda_visible = ",".join(str(gpu) for gpu in assigned_gpus)
            
            return assigned_gpus, cuda_visible
    
    def release_resources(self, module_name: str):
        """
        Release resources allocated to a module.
        
        Args:
            module_name: Name of the module
        """
        with self._lock:
            if module_name not in self._allocated_to_modules:
                return
            
            allocation = self._allocated_to_modules[module_name]
            gpus = allocation.get("gpus", [])
            
            # Release GPUs
            for gpu in gpus:
                self._gpu_in_use.discard(gpu)
            
            # Remove allocation
            del self._allocated_to_modules[module_name]
    
    def borrow_resources(
        self,
        task_id: str,
        cpus: int,
        gpus: int
    ) -> Tuple[List[int], Optional[str]]:
        """
        Borrow resources for nested task execution.
        
        Args:
            task_id: Unique identifier for the nested task
            cpus: Number of CPUs required
            gpus: Number of GPUs required
            
        Returns:
            Tuple of (assigned_gpu_ids, cuda_visible_devices_string)
            
        Raises:
            ResourceError: If resources are not available
        """
        with self._lock:
            # Check if task already has resources borrowed
            if task_id in self._borrowed_for_nested:
                raise ResourceError(f"Task '{task_id}' already has resources borrowed")
            
            # Validate resource availability
            if cpus > self.available_cpus:
                raise ResourceError(
                    f"Insufficient CPUs for nested task: requested {cpus}, available {self.available_cpus}"
                )
            
            if gpus > self.available_gpus:
                raise ResourceError(
                    f"Insufficient GPUs for nested task: requested {gpus}, available {self.available_gpus}"
                )
            
            # Allocate GPUs (assign specific devices)
            assigned_gpus = []
            if gpus > 0:
                available_gpu_list = [gpu for gpu in self._available_gpus if gpu not in self._gpu_in_use]
                # Also exclude GPUs already borrowed
                borrowed_gpus = set()
                for alloc in self._borrowed_for_nested.values():
                    borrowed_gpus.update(alloc.get("gpus", []))
                available_gpu_list = [gpu for gpu in available_gpu_list if gpu not in borrowed_gpus]
                
                if len(available_gpu_list) < gpus:
                    raise ResourceError(
                        f"Insufficient GPU devices for nested task: requested {gpus}, "
                        f"available {len(available_gpu_list)}"
                    )
                
                assigned_gpus = available_gpu_list[:gpus]
            
            # Record borrowing
            self._borrowed_for_nested[task_id] = {
                "cpus": cpus,
                "gpus": assigned_gpus
            }
            
            # Create CUDA_VISIBLE_DEVICES string
            cuda_visible = None
            if assigned_gpus:
                cuda_visible = ",".join(str(gpu) for gpu in assigned_gpus)
            
            return assigned_gpus, cuda_visible
    
    def return_borrowed_resources(self, task_id: str):
        """
        Return borrowed resources from nested task execution.
        
        Args:
            task_id: Unique identifier for the nested task
        """
        with self._lock:
            if task_id not in self._borrowed_for_nested:
                return
            
            # Note: We don't track borrowed GPUs in _gpu_in_use
            # They're tracked separately in _borrowed_for_nested
            # So we just remove the record
            del self._borrowed_for_nested[task_id]
    
    def validate_resources(self, cpus: int, gpus: int) -> bool:
        """
        Validate if resources are available (without reserving).
        
        Args:
            cpus: Number of CPUs required
            gpus: Number of GPUs required
            
        Returns:
            True if resources are available, False otherwise
        """
        return cpus <= self.available_cpus and gpus <= self.available_gpus
    
    def get_module_resources(self, module_name: str) -> Optional[Dict]:
        """
        Get resources allocated to a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary with 'cpus' and 'gpus' keys, or None if not allocated
        """
        with self._lock:
            return self._allocated_to_modules.get(module_name)
    
    def get_status(self) -> Dict:
        """
        Get current resource allocation status.
        
        Returns:
            Dictionary with resource status information
        """
        with self._lock:
            return {
                "total_cpus": self.total_cpus,
                "total_gpus": self.total_gpus,
                "allocated_cpus": self.allocated_cpus,
                "allocated_gpus": self.allocated_gpus,
                "borrowed_cpus": self.borrowed_cpus,
                "borrowed_gpus": self.borrowed_gpus,
                "available_cpus": self.available_cpus,
                "available_gpus": self.available_gpus,
                "allocated_modules": list(self._allocated_to_modules.keys()),
                "borrowed_tasks": list(self._borrowed_for_nested.keys())
            }

