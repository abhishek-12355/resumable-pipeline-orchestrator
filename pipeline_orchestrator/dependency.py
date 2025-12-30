"""Dependency graph management for pipeline orchestrator."""

import time
from typing import Dict, List, Set, Optional
from collections import defaultdict, deque

from pipeline_orchestrator.exceptions import DependencyError, ConfigurationError


class DependencyGraph:
    """Manages module dependency graph and execution order."""
    
    def __init__(self, config):
        """
        Initialize dependency graph from configuration.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self.modules = {m["name"]: m for m in config.modules}
        
        # Build adjacency list: module -> list of dependencies
        self.dependencies: Dict[str, List[str]] = defaultdict(list)
        
        # Build reverse adjacency: module -> list of dependents
        self.dependents: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize graph
        self._build_graph()
        
        # Validate no cycles
        self._validate_no_cycles()
        
        # Track completed modules (for resumption)
        self.completed_modules: Set[str] = set()
        
        # Track dependency completion times for queue ordering
        # Maps module_name -> timestamp when last dependency completed
        self._dependency_completion_times: Dict[str, float] = {}
    
    def _build_graph(self):
        """Build dependency graph from configuration."""
        for module_name, module_config in self.modules.items():
            depends_on = module_config.get("depends_on", [])
            self.dependencies[module_name] = depends_on
            
            for dep in depends_on:
                self.dependents[dep].append(module_name)
    
    def _validate_no_cycles(self):
        """Validate that dependency graph has no cycles."""
        # Kahn's algorithm to detect cycles
        in_degree = {module: 0 for module in self.modules}
        
        for module, deps in self.dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[module] += 1
        
        queue = deque([m for m, degree in in_degree.items() if degree == 0])
        processed = 0
        
        while queue:
            module = queue.popleft()
            processed += 1
            
            for dependent in self.dependents[module]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if processed != len(self.modules):
            raise ConfigurationError("Circular dependency detected in module graph")
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order using topological sort.
        Returns list of batches, where each batch can be executed in parallel.
        
        Returns:
            List of batches, each batch is a list of module names that can run in parallel
        """
        # Kahn's algorithm for topological sort with batches
        in_degree = {module: len(self.dependencies[module]) for module in self.modules}
        
        # Exclude completed modules
        active_in_degree = {
            m: degree for m, degree in in_degree.items()
            if m not in self.completed_modules
        }
        
        # Adjust in-degree for dependencies on completed modules
        for module in active_in_degree:
            # Count only non-completed dependencies
            active_deps = [
                dep for dep in self.dependencies[module]
                if dep not in self.completed_modules
            ]
            active_in_degree[module] = len(active_deps)
        
        batches = []
        queue = deque([m for m, degree in active_in_degree.items() if degree == 0])
        
        while queue:
            batch = list(queue)
            batches.append(batch)
            queue.clear()
            
            for module in batch:
                for dependent in self.dependents[module]:
                    if dependent in active_in_degree:
                        active_in_degree[dependent] -= 1
                        if active_in_degree[dependent] == 0:
                            queue.append(dependent)
        
        return batches
    
    def get_dependencies(self, module_name: str) -> List[str]:
        """
        Get list of dependencies for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of dependency module names
        """
        if module_name not in self.modules:
            raise DependencyError(f"Module '{module_name}' not found")
        
        return self.dependencies[module_name]
    
    def get_dependents(self, module_name: str) -> List[str]:
        """
        Get list of modules that depend on this module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of dependent module names
        """
        if module_name not in self.modules:
            raise DependencyError(f"Module '{module_name}' not found")
        
        return self.dependents[module_name]
    
    def get_ready_modules(self) -> List[str]:
        """
        Get list of modules that are ready to execute (all dependencies satisfied).
        
        Returns:
            List of module names ready to execute
        """
        ready = []
        
        for module_name in self.modules:
            if module_name in self.completed_modules:
                continue
            
            # Check if all dependencies are completed
            deps = self.dependencies[module_name]
            if all(dep in self.completed_modules for dep in deps):
                ready.append(module_name)
        
        return ready
    
    def mark_completed(self, module_name: str):
        """
        Mark a module as completed.
        
        Args:
            module_name: Name of the completed module
        """
        if module_name not in self.modules:
            raise DependencyError(f"Module '{module_name}' not found")
        
        self.completed_modules.add(module_name)
        
        # Update dependency completion times for dependents
        completion_time = time.time()
        for dependent in self.dependents[module_name]:
            # Update the dependent's dependency completion time
            # Use the latest completion time among all dependencies
            current_time = self._dependency_completion_times.get(dependent, 0.0)
            self._dependency_completion_times[dependent] = max(current_time, completion_time)
    
    def is_completed(self, module_name: str) -> bool:
        """
        Check if a module is completed.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if module is completed, False otherwise
        """
        return module_name in self.completed_modules
    
    def all_completed(self) -> bool:
        """
        Check if all modules are completed.
        
        Returns:
            True if all modules are completed, False otherwise
        """
        return len(self.completed_modules) == len(self.modules)
    
    def get_module_config(self, module_name: str) -> Dict:
        """
        Get configuration for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module configuration dictionary
        """
        if module_name not in self.modules:
            raise DependencyError(f"Module '{module_name}' not found")
        
        return self.modules[module_name]
    
    def get_ready_modules_with_dependency_order(self) -> List[tuple]:
        """
        Get ready modules ordered by dependency completion time.
        
        Returns:
            List of tuples (module_name, dependency_completion_time)
            Ordered by dependency_completion_time (ascending), then by module name
        """
        ready_modules = []
        
        for module_name in self.modules:
            if module_name in self.completed_modules:
                continue
            
            # Check if all dependencies are completed
            deps = self.dependencies[module_name]
            if all(dep in self.completed_modules for dep in deps):
                # Get dependency completion time (when last dependency completed)
                completion_time = self._dependency_completion_times.get(module_name, 0.0)
                ready_modules.append((module_name, completion_time))
        
        # Sort by dependency_completion_time (ascending), then by module name
        ready_modules.sort(key=lambda x: (x[1], x[0]))
        
        return ready_modules
    
    def get_dependency_completion_time(self, module_name: str) -> float:
        """
        Get the dependency completion time for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Timestamp when last dependency completed, or 0.0 if no dependencies or not tracked
        """
        return self._dependency_completion_times.get(module_name, 0.0)
    
    def reset_completed_modules(self):
        """
        Reset all completed modules and dependency completion times.
        
        This clears the completion tracking state, allowing all modules
        to be re-executed. Useful for force restart scenarios.
        """
        self.completed_modules.clear()
        self._dependency_completion_times.clear()

