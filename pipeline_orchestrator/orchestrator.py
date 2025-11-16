"""Main pipeline orchestrator."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline_orchestrator.config import PipelineConfig
from pipeline_orchestrator.dependency import DependencyGraph
from pipeline_orchestrator.resources import ResourceManager
from pipeline_orchestrator.checkpoint import PipelineCheckpointManager
from pipeline_orchestrator.results import ResultsManager
from pipeline_orchestrator.executor import SequentialExecutor, ParallelExecutor
from pipeline_orchestrator.module import ModuleLoader, BaseModule
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.exceptions import (
    ConfigurationError,
    ResourceError,
    ModuleExecutionError
)
from pipeline_orchestrator.ipc import WorkerIPCManager


class PipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_path: Path to YAML configuration file
            config: PipelineConfig instance (if not using config_path)
        """
        # Load configuration
        if config is None:
            if config_path is None:
                raise ConfigurationError("Either config_path or config must be provided")
            self.config = PipelineConfig.from_yaml_file(config_path)
        else:
            self.config = config
        
        # Initialize components
        self.dependency_graph = DependencyGraph(self.config)
        
        # Resource manager
        self.resource_manager = ResourceManager(
            max_cpus=self.config.resources.get("max_cpus"),
            max_gpus=self.config.resources.get("max_gpus")
        )
        
        # Checkpoint manager
        checkpoint_config = self.config.checkpoint
        self.checkpoint_manager = PipelineCheckpointManager(
            checkpoint_directory=checkpoint_config.get("directory", "./.checkpoints"),
            enabled=checkpoint_config.get("enabled", True)
        ) if checkpoint_config.get("enabled", True) else None
        
        # Results manager
        self.results_manager = ResultsManager(
            dependency_graph=self.dependency_graph,
            checkpoint_manager=self.checkpoint_manager
        )
        
        # IPC manager (for nested execution in process mode)
        self.ipc_manager = WorkerIPCManager() if self.config.execution["worker_type"] == "process" else None
        
        # Executor
        execution_config = self.config.execution
        if execution_config["mode"] == "sequential":
            self.executor = SequentialExecutor(self.resource_manager)
        else:
            self.executor = ParallelExecutor(
                resource_manager=self.resource_manager,
                worker_type=execution_config["worker_type"],
                failure_policy=execution_config["failure_policy"],
                max_nested_depth=execution_config.get("max_nested_depth"),
                ipc_manager=self.ipc_manager
            )
        
        # Module cache
        self._modules: Dict[str, BaseModule] = {}
        
        # Execution state
        self._execution_results: Dict[str, Any] = {}
    
    def load_modules(self):
        """Load all modules from configuration."""
        for module_config in self.config.modules:
            module_name = module_config["name"]
            if module_name not in self._modules:
                module = ModuleLoader.load_module(module_config)
                self._modules[module_name] = module
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Returns:
            Dictionary mapping module names to results (or errors)
        """
        # Load modules
        self.load_modules()
        
        # Get execution order (batches of parallelizable modules)
        execution_batches = self.dependency_graph.get_execution_order()
        
        # Execute batches sequentially, modules in batch in parallel
        all_results = {}
        
        for batch in execution_batches:
            # Skip already completed modules
            batch = [m for m in batch if not self.dependency_graph.is_completed(m)]
            if not batch:
                continue
            
            # Prepare modules and contexts for this batch
            batch_modules = {}
            batch_contexts = {}
            
            for module_name in batch:
                # Get module
                module = self._modules[module_name]
                
                # Get module config
                module_config = self.config.get_module_config(module_name)
                if module_config is None:
                    raise ConfigurationError(f"Module config not found: {module_name}")
                
                # Reserve resources
                resources = module_config.get("resources", {})
                cpus = resources.get("cpus", 1)
                gpus = resources.get("gpus", 0)
                
                try:
                    assigned_gpus, cuda_visible = self.resource_manager.reserve_resources(
                        module_name, cpus, gpus
                    )
                except ResourceError as e:
                    if self.config.execution["failure_policy"] == "fail_fast":
                        raise
                    # In collect_all mode, mark as error and continue
                    all_results[module_name] = e
                    continue
                
                # Get PyTorch device strings and GPU names for allocated GPUs
                pytorch_devices = []
                gpu_names = []
                for gpu_id in assigned_gpus:
                    pytorch_device = self.resource_manager.get_pytorch_device(gpu_id)
                    pytorch_devices.append(pytorch_device)
                    gpu_name = self.resource_manager.get_gpu_name(gpu_id)
                    if gpu_name:
                        gpu_names.append(gpu_name)
                    else:
                        gpu_names.append(f"GPU {gpu_id}")
                
                # Create context
                dependency_results = self.results_manager.get_dependency_results(module_name)
                
                # Create nested execution function for context
                def create_execute_tasks_fn(module_name: str):
                    def execute_tasks(tasks):
                        return self._execute_nested_tasks(module_name, tasks)
                    return execute_tasks
                
                context = ModuleContext(
                    module_name=module_name,
                    pipeline_name=self.config.name,
                    resources={
                        "cpus": cpus,
                        "gpus": gpus,
                        "gpu_ids": assigned_gpus,
                        "cuda_visible_devices": cuda_visible,
                        "pytorch_devices": pytorch_devices,
                        "gpu_names": gpu_names
                    },
                    dependency_results=dependency_results,
                    execute_tasks_fn=create_execute_tasks_fn(module_name)
                )
                
                batch_modules[module_name] = module
                batch_contexts[module_name] = context
            
            # Execute batch
            batch_results = self.executor.execute_modules(
                batch_modules,
                batch_contexts,
                self.results_manager
            )
            
            # Release resources
            for module_name in batch:
                self.resource_manager.release_resources(module_name)
            
            # Collect results
            all_results.update(batch_results)
            
            # Check for failures in fail-fast mode
            if self.config.execution["failure_policy"] == "fail_fast":
                for module_name, result in batch_results.items():
                    if isinstance(result, Exception):
                        # Stop execution on first failure
                        return all_results
        
        self._execution_results = all_results
        return all_results
    
    def _execute_nested_tasks(self, requesting_module_name: str, tasks: List[Any]) -> List[Any]:
        """
        Execute nested tasks requested by a module.
        
        Args:
            requesting_module_name: Name of the module requesting nested execution
            tasks: List of tasks to execute
            
        Returns:
            List of results/errors in same order as tasks
        """
        if not tasks:
            return []
        
        # Validate all tasks are callable
        for i, task in enumerate(tasks):
            if not callable(task):
                from pipeline_orchestrator.exceptions import NestedExecutionError
                return [NestedExecutionError(f"Task {i} is not callable: {task}")] * len(tasks)
        
        # Get execution configuration
        execution_config = self.config.execution
        worker_type = execution_config["worker_type"]
        max_nested_depth = execution_config.get("max_nested_depth")
        
        # Validate nested depth (if limited)
        # Note: We don't track depth currently, but could be added
        if max_nested_depth is not None and max_nested_depth <= 0:
            from pipeline_orchestrator.exceptions import NestedExecutionError
            return [NestedExecutionError("Maximum nested execution depth exceeded")] * len(tasks)
        
        # Execute tasks using executor's nested execution method
        # For now, execute tasks sequentially to avoid resource conflicts
        # In a full implementation, we would:
        # 1. Check available resources
        # 2. Reserve resources for nested tasks
        # 3. Execute tasks in parallel based on worker_type
        # 4. Release resources
        # 5. Return results
        
        # For nested tasks, use simple parallel execution via executor
        # The executor handles resource management
        if isinstance(self.executor, ParallelExecutor):
            # Use executor's nested task execution
            return self.executor._execute_nested_tasks(tasks, self.results_manager)
        else:
            # Sequential executor - execute tasks sequentially
            results = []
            for task in tasks:
                try:
                    result = task()
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get execution results.
        
        Returns:
            Dictionary mapping module names to results (or errors)
        """
        return self._execution_results.copy()
    
    def get_module_result(self, module_name: str) -> Any:
        """
        Get result for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module result or error
        """
        return self._execution_results.get(module_name)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.ipc_manager:
            self.ipc_manager.cleanup()

