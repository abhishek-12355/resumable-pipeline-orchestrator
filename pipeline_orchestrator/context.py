"""Execution context for modules."""

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
            resources: Resource allocation (cpus, gpus, gpu_ids)
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

