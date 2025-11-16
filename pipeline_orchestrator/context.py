"""Execution context for modules."""

import subprocess
from typing import Any, Callable, Dict, List, Optional

from pipeline_orchestrator.exceptions import DependencyError


class ModuleContext:
    """Execution context provided to modules during execution."""
    
    def __init__(
        self,
        module_name: str,
        pipeline_name: str,
        resources: Dict[str, Any],
        dependency_results: Dict[str, Any],
        execute_tasks_fn: Optional[Callable] = None
    ):
        """
        Initialize module context.
        
        Args:
            module_name: Name of the current module
            pipeline_name: Name of the pipeline
            resources: Resource allocation (cpus, gpus, gpu_ids, cuda_visible_devices, pytorch_devices, gpu_names)
            dependency_results: Dictionary of dependency results
            execute_tasks_fn: Function to execute nested tasks (optional)
        """
        self.module_name = module_name
        self.pipeline_name = pipeline_name
        self.resources = resources
        self._dependency_results = dependency_results
        self._execute_tasks_fn = execute_tasks_fn
    
    @property
    def dependency_results(self) -> Dict[str, Any]:
        """
        Get dictionary of all dependency results.
        
        Returns:
            Dictionary mapping dependency module names to results
        """
        return self._dependency_results.copy()
    
    def get_result(self, module_name: str) -> Any:
        """
        Get result from a specific dependency module.
        
        Args:
            module_name: Name of the dependency module
            
        Returns:
            Result object from the dependency module
            
        Raises:
            DependencyError: If module not found or not completed
        """
        if module_name not in self._dependency_results:
            raise DependencyError(
                f"Module '{module_name}' is not a dependency of '{self.module_name}' "
                f"or has not been completed"
            )
        
        return self._dependency_results[module_name]
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        Get all completed module results (not just dependencies).
        
        Note: This method is only available if the execute_tasks_fn provides
        access to the results manager.
        
        Returns:
            Dictionary mapping all completed module names to results
        """
        # This requires access to results manager, which is available
        # through execute_tasks_fn if needed
        # For now, return dependency results only
        # Full implementation requires access to ResultsManager
        return self._dependency_results.copy()
    
    def execute_tasks(self, tasks: List[Any]) -> List[Any]:
        """
        Execute tasks in parallel via orchestrator.
        
        This method allows a module to request parallel execution of sub-tasks.
        The orchestrator handles resource management and execution.
        
        Args:
            tasks: List of tasks to execute in parallel
            
        Returns:
            List of results/errors in same order as tasks.
            Each result can be:
            - Success: The result data
            - Error: Exception object
            
        Raises:
            NestedExecutionError: If nested execution is not available or fails
        """
        if self._execute_tasks_fn is None:
            from pipeline_orchestrator.exceptions import NestedExecutionError
            raise NestedExecutionError(
                "Nested execution is not available in this context"
            )
        
        return self._execute_tasks_fn(tasks)
    
    def get_pytorch_device(self, gpu_index: int = 0) -> str:
        """
        Get PyTorch device string for the allocated GPU.
        
        For NVIDIA GPUs: returns "cuda:0", "cuda:1", etc.
        For Apple Silicon/Metal GPUs: returns "mps:0"
        If no GPU allocated or invalid index: returns "cpu"
        
        Args:
            gpu_index: GPU index within allocated GPUs (default: 0, first GPU)
            
        Returns:
            PyTorch device string (e.g., "cuda:0", "mps:0", "cpu")
        """
        # Get PyTorch device strings from resources
        pytorch_devices = self.resources.get("pytorch_devices", [])
        
        if not pytorch_devices:
            return "cpu"
        
        if gpu_index < 0 or gpu_index >= len(pytorch_devices):
            # Use first GPU if index out of range
            gpu_index = 0
        
        return pytorch_devices[gpu_index]
    
    def get_gpu_name(self, gpu_index: int = 0) -> Optional[str]:
        """
        Get GPU name/model for the allocated GPU.
        
        Args:
            gpu_index: GPU index within allocated GPUs (default: 0, first GPU)
            
        Returns:
            GPU name/model string or None if not available/not found
        """
        # Get GPU names from resources
        gpu_names = self.resources.get("gpu_names", [])
        
        if not gpu_names:
            return None
        
        if gpu_index < 0 or gpu_index >= len(gpu_names):
            gpu_index = 0
        
        return gpu_names[gpu_index]

