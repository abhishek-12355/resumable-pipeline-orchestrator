"""Resource management for CPU and GPU allocation."""

import os
import subprocess
import threading
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from pipeline_orchestrator.exceptions import ResourceError
from pipeline_orchestrator.logging_config import get_logger

logger = get_logger(__name__)


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
        
        logger.info(
            f"ResourceManager initialized: {self.max_cpus} CPUs, {self.max_gpus} GPUs "
            f"(max_cpus={max_cpus}, max_gpus={max_gpus})"
        )
        
        # Resource allocation tracking
        self._lock = threading.RLock()
        
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
            count = psutil.cpu_count(logical=True)
            logger.debug(f"Detected {count} CPUs using psutil")
            return count
        except ImportError:
            # Fallback to os.cpu_count()
            count = os.cpu_count()
            count = count if count else 1
            logger.debug(f"Detected {count} CPUs using os.cpu_count()")
            return count
    
    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        # Try nvidia-smi first (NVIDIA GPUs)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_list = result.stdout.strip().split('\n')
                if gpu_list and gpu_list[0]:  # Check if not empty
                    count = len(gpu_list)
                    logger.debug(f"Detected {count} GPU(s) using nvidia-smi")
                    return count
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            logger.debug("nvidia-smi not available or failed")
            pass
        
        # Try pynvml as fallback (NVIDIA GPUs)
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            if count > 0:
                logger.debug(f"Detected {count} GPU(s) using pynvml")
                return count
        except (ImportError, Exception):
            logger.debug("pynvml not available or failed")
            pass
        
        # Try Apple Silicon/Metal GPU detection (macOS)
        try:
            import platform
            if platform.system() == "Darwin":  # macOS
                # Use system_profiler to detect Metal GPUs
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Count GPU entries (look for "Chipset Model" or "Metal" keywords)
                    output = result.stdout
                    # Look for Metal-capable GPUs
                    # Apple Silicon always has integrated GPU
                    # Intel Macs may have discrete GPUs
                    if "Metal:" in output or "Chipset Model:" in output:
                        # Count unique GPU chipsets
                        # For Apple Silicon, typically 1 GPU (unified memory)
                        # For Intel Macs, may have integrated + discrete
                        gpu_count = output.count("Chipset Model:")
                        # Metal should be supported
                        if "Metal:" in output and gpu_count == 0:
                            # No chipset info but Metal present - likely Apple Silicon
                            gpu_count = 1
                        if gpu_count > 0:
                            return gpu_count
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        except Exception:
            pass
        
        # Try checking for Metal framework (macOS specific)
        try:
            import platform
            if platform.system() == "Darwin":
                # On macOS, assume 1 GPU if we're on Apple Silicon
                # or check Metal availability
                result = subprocess.run(
                    ["uname", "-m"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and "arm64" in result.stdout:
                    # Apple Silicon - always has integrated GPU
                    return 1
        except Exception:
            pass
        
        # No GPUs detected
        logger.debug("No GPUs detected")
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
                logger.warning(f"Module '{module_name}' already has resources allocated")
                raise ResourceError(f"Module '{module_name}' already has resources allocated")
            
            # Validate resource availability
            if cpus > self.available_cpus:
                logger.error(
                    f"Insufficient CPUs for {module_name}: requested {cpus}, "
                    f"available {self.available_cpus}"
                )
                raise ResourceError(
                    f"Insufficient CPUs: requested {cpus}, available {self.available_cpus}"
                )
            
            if gpus > self.available_gpus:
                logger.error(
                    f"Insufficient GPUs for {module_name}: requested {gpus}, "
                    f"available {self.available_gpus}"
                )
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
            
            logger.debug(
                f"Reserved resources for {module_name}: {cpus} CPUs, "
                f"{len(assigned_gpus)} GPUs {assigned_gpus if assigned_gpus else ''}"
            )
            
            return assigned_gpus, cuda_visible
    
    def try_reserve_resources(
        self,
        module_name: str,
        cpus: int,
        gpus: int
    ) -> Tuple[bool, List[int], Optional[str]]:
        """
        Try to reserve resources for a module (non-blocking, doesn't raise).
        
        Args:
            module_name: Name of the module
            cpus: Number of CPUs required
            gpus: Number of GPUs required
            
        Returns:
            Tuple of (success: bool, assigned_gpu_ids: List[int], cuda_visible_devices_string: Optional[str])
            If success is False, gpu_ids will be empty list and cuda_visible will be None
        """
        try:
            assigned_gpus, cuda_visible = self.reserve_resources(module_name, cpus, gpus)
            return True, assigned_gpus, cuda_visible
        except ResourceError:
            return False, [], None
    
    def release_resources(self, module_name: str):
        """
        Release resources allocated to a module.
        
        Args:
            module_name: Name of the module
        """
        with self._lock:
            if module_name not in self._allocated_to_modules:
                logger.debug(f"Module {module_name} has no resources to release")
                return
            
            allocation = self._allocated_to_modules[module_name]
            gpus = allocation.get("gpus", [])
            cpus = allocation.get("cpus", 0)
            
            # Release GPUs
            for gpu in gpus:
                self._gpu_in_use.discard(gpu)
            
            # Remove allocation
            del self._allocated_to_modules[module_name]
            
            logger.debug(
                f"Released resources for {module_name}: {cpus} CPUs, "
                f"{len(gpus)} GPUs {gpus if gpus else ''}"
            )
    
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
                logger.warning(f"Task '{task_id}' already has resources borrowed")
                raise ResourceError(f"Task '{task_id}' already has resources borrowed")
            
            # Validate resource availability
            if cpus > self.available_cpus:
                logger.error(
                    f"Insufficient CPUs for nested task {task_id}: requested {cpus}, "
                    f"available {self.available_cpus}"
                )
                raise ResourceError(
                    f"Insufficient CPUs for nested task: requested {cpus}, available {self.available_cpus}"
                )
            
            if gpus > self.available_gpus:
                logger.error(
                    f"Insufficient GPUs for nested task {task_id}: requested {gpus}, "
                    f"available {self.available_gpus}"
                )
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
            
            logger.debug(
                f"Borrowed resources for nested task {task_id}: {cpus} CPUs, "
                f"{len(assigned_gpus)} GPUs {assigned_gpus if assigned_gpus else ''}"
            )
            
            return assigned_gpus, cuda_visible
    
    def return_borrowed_resources(self, task_id: str):
        """
        Return borrowed resources from nested task execution.
        
        Args:
            task_id: Unique identifier for the nested task
        """
        with self._lock:
            if task_id not in self._borrowed_for_nested:
                logger.debug(f"Nested task {task_id} has no borrowed resources to return")
                return
            
            allocation = self._borrowed_for_nested[task_id]
            cpus = allocation.get("cpus", 0)
            gpus = allocation.get("gpus", [])
            
            # Note: We don't track borrowed GPUs in _gpu_in_use
            # They're tracked separately in _borrowed_for_nested
            # So we just remove the record
            del self._borrowed_for_nested[task_id]
            
            logger.debug(
                f"Returned borrowed resources for nested task {task_id}: {cpus} CPUs, "
                f"{len(gpus)} GPUs {gpus if gpus else ''}"
            )
    
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
    
    def get_gpu_name(self, gpu_index: int = 0) -> Optional[str]:
        """
        Get GPU name/model for a specific GPU index.
        
        Args:
            gpu_index: GPU index (0-indexed)
            
        Returns:
            GPU name/model string or None if not available/not found
        """
        if gpu_index < 0 or gpu_index >= self.max_gpus:
            return None
        
        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={gpu_index}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                if gpu_name:
                    return gpu_name
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Try pynvml for NVIDIA GPUs
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            pynvml.nvmlShutdown()
            if gpu_name:
                return gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else gpu_name
        except (ImportError, Exception):
            pass
        
        # Try Apple Silicon/Metal GPU (macOS)
        try:
            import platform
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    output = result.stdout
                    # Look for "Chipset Model:" line
                    lines = output.split('\n')
                    chipset_models = []
                    for i, line in enumerate(lines):
                        if "Chipset Model:" in line:
                            model = line.split("Chipset Model:")[-1].strip()
                            if model:
                                chipset_models.append(model)
                    
                    # For Apple Silicon, typically 1 GPU
                    if chipset_models and gpu_index < len(chipset_models):
                        return chipset_models[gpu_index]
                    elif chipset_models:
                        # Return first model if index out of range
                        return chipset_models[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        except Exception:
            pass
        
        return None
    
    def get_pytorch_device(self, gpu_index: int = 0) -> str:
        """
        Get PyTorch device string for a GPU index.
        
        For NVIDIA GPUs: returns "cuda:0", "cuda:1", etc.
        For Apple Silicon/Metal GPUs: returns "mps:0"
        If no GPU or invalid index: returns "cpu"
        
        Args:
            gpu_index: GPU index (0-indexed)
            
        Returns:
            PyTorch device string (e.g., "cuda:0", "mps:0", "cpu")
        """
        if gpu_index < 0 or gpu_index >= self.max_gpus:
            return "cpu"
        
        # Check if NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # NVIDIA GPU
                return f"cuda:{gpu_index}"
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Check if Apple Silicon/Metal GPU
        try:
            import platform
            if platform.system() == "Darwin":
                # Check for Apple Silicon (arm64)
                result = subprocess.run(
                    ["uname", "-m"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and "arm64" in result.stdout:
                    # Apple Silicon with Metal Performance Shaders
                    return f"mps:{gpu_index}"
        except Exception:
            pass
        
        # Default to CPU
        return "cpu"
    
    def get_all_gpu_names(self) -> List[str]:
        """
        Get names/models for all available GPUs.
        
        Returns:
            List of GPU names/models (one per GPU)
        """
        gpu_names = []
        for i in range(self.max_gpus):
            name = self.get_gpu_name(i)
            if name:
                gpu_names.append(name)
            else:
                gpu_names.append(f"GPU {i}")
        return gpu_names
    
    def get_all_pytorch_devices(self) -> List[str]:
        """
        Get PyTorch device strings for all available GPUs.
        
        Returns:
            List of PyTorch device strings (e.g., ["cuda:0", "cuda:1"] or ["mps:0"])
        """
        devices = []
        for i in range(self.max_gpus):
            device = self.get_pytorch_device(i)
            if device != "cpu":
                devices.append(device)
        return devices

