"""Results management and passing for pipeline orchestrator."""

from typing import Any, Dict, Optional

from pipeline_orchestrator.exceptions import DependencyError
from pipeline_orchestrator.checkpoint import PipelineCheckpointManager
from pipeline_orchestrator.dependency import DependencyGraph


class ResultsManager:
    """Manages module results in memory and via checkpoints."""
    
    def __init__(
        self,
        dependency_graph: DependencyGraph,
        checkpoint_manager: Optional[PipelineCheckpointManager]
    ):
        """
        Initialize results manager.
        
        Args:
            dependency_graph: Dependency graph instance
            checkpoint_manager: Checkpoint manager instance (can be None)
        """
        self.dependency_graph = dependency_graph
        self.checkpoint_manager = checkpoint_manager
        
        # In-memory results storage: {module_name: result}
        self._results: Dict[str, Any] = {}
        
        # Load completed modules from checkpoints on initialization
        self._load_completed_from_checkpoints()
    
    def _load_completed_from_checkpoints(self):
        """Load completed modules from checkpoints on startup."""
        if not self.checkpoint_manager or not self.checkpoint_manager.enabled:
            return
        
        completed_modules = self.checkpoint_manager.list_completed_modules()
        
        for module_name in completed_modules:
            # Mark as completed in dependency graph
            if not self.dependency_graph.is_completed(module_name):
                try:
                    # Load result from checkpoint
                    result, _ = self.checkpoint_manager.load_result(module_name)
                    self._results[module_name] = result
                    self.dependency_graph.mark_completed(module_name)
                except Exception:
                    # Skip if checkpoint load fails
                    pass
    
    def save_result(self, module_name: str, result: Any, is_error: bool = False):
        """
        Save module result (in-memory and checkpoint).
        
        Args:
            module_name: Name of the module
            result: Result object or error to save
            is_error: Whether result is an error/exception
        """
        # Store in memory (even errors)
        self._results[module_name] = result
        
        # Save to checkpoint if enabled (always checkpoint, even failures)
        if self.checkpoint_manager and self.checkpoint_manager.enabled:
            try:
                self.checkpoint_manager.save_result(module_name, result, is_error=is_error)
            except Exception:
                # Log but don't fail if checkpoint save fails
                pass
        
        # Mark as completed in dependency graph (even if failed)
        # This allows downstream modules to know the module was attempted
        self.dependency_graph.mark_completed(module_name)
    
    def get_result(self, module_name: str) -> Any:
        """
        Get result for a module (from memory or checkpoint).
        
        Args:
            module_name: Name of the module
            
        Returns:
            Result object
            
        Raises:
            DependencyError: If module not found or not completed
        """
        # First try in-memory
        if module_name in self._results:
            return self._results[module_name]
        
        # Try checkpoint if enabled
        if self.checkpoint_manager and self.checkpoint_manager.enabled:
            if self.checkpoint_manager.has_checkpoint(module_name):
                try:
                    result, _ = self.checkpoint_manager.load_result(module_name)
                    # Cache in memory for future access
                    self._results[module_name] = result
                    return result
                except Exception as e:
                    raise DependencyError(
                        f"Failed to load checkpoint for module '{module_name}': {e}"
                    )
        
        # Check if module exists in dependency graph
        try:
            self.dependency_graph.get_module_config(module_name)
        except DependencyError:
            raise DependencyError(f"Module '{module_name}' not found")
        
        # Module exists but not completed
        raise DependencyError(
            f"Module '{module_name}' has not been executed or checkpointed"
        )
    
    def get_dependency_results(self, module_name: str) -> Dict[str, Any]:
        """
        Get results for all direct dependencies of a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary mapping dependency names to results
        """
        dependencies = self.dependency_graph.get_dependencies(module_name)
        dependency_results = {}
        
        for dep_name in dependencies:
            try:
                dependency_results[dep_name] = self.get_result(dep_name)
            except DependencyError:
                # Skip if dependency not available
                pass
        
        return dependency_results
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        Get all completed module results.
        
        Returns:
            Dictionary mapping module names to results
        """
        # Get all results from memory
        all_results = self._results.copy()
        
        # If checkpoint manager enabled, also check for any missing checkpoints
        if self.checkpoint_manager and self.checkpoint_manager.enabled:
            completed_modules = self.checkpoint_manager.list_completed_modules()
            for module_name in completed_modules:
                if module_name not in all_results:
                    try:
                        result, _ = self.checkpoint_manager.load_result(module_name)
                        all_results[module_name] = result
                    except Exception:
                        # Skip if load fails
                        pass
        
        return all_results
    
    def has_result(self, module_name: str) -> bool:
        """
        Check if result exists for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if result exists, False otherwise
        """
        # Check memory first
        if module_name in self._results:
            return True
        
        # Check checkpoint
        if self.checkpoint_manager and self.checkpoint_manager.enabled:
            return self.checkpoint_manager.has_checkpoint(module_name)
        
        return False
    
    def is_completed(self, module_name: str) -> bool:
        """
        Check if module is completed.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if module is completed, False otherwise
        """
        return self.dependency_graph.is_completed(module_name)
    
    def clear_result(self, module_name: str):
        """
        Clear result for a module (memory only, checkpoint remains).
        
        Args:
            module_name: Name of the module
        """
        if module_name in self._results:
            del self._results[module_name]

