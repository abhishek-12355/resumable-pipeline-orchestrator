"""Main pipeline orchestrator."""

import glob
import sys
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from pipeline_orchestrator.checkpoint import PipelineCheckpointManager
from pipeline_orchestrator.config import PipelineConfig
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.dependency import DependencyGraph
from pipeline_orchestrator.exceptions import (
    ConfigurationError,
    ModuleExecutionError,
    ResourceError,
)
from pipeline_orchestrator.executor import ParallelExecutor, SequentialExecutor
from pipeline_orchestrator.ipc import WorkerIPCManager
from pipeline_orchestrator.logging import ModuleLogManager
from pipeline_orchestrator.logging_config import get_logger
from pipeline_orchestrator.module import BaseModule, ModuleLoader
from pipeline_orchestrator.resources import ResourceManager
from pipeline_orchestrator.results import ResultsManager
from pipeline_orchestrator.ui import LiveDashboard
from pipeline_orchestrator.queue import ModuleQueue, ExecutionQueueManager

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_path: Path to YAML configuration file
            config: PipelineConfig instance (if not using config_path)
        """
        global logger
        # Load configuration
        if config is None:
            if config_path is None:
                raise ConfigurationError("Either config_path or config must be provided")
            self.config = PipelineConfig.from_yaml_file(config_path)
        else:
            self.config = config


        logging_cfg = self.config.logging or {}
        self.enable_live_logs = logging_cfg.get("enable_live_logs", True)
        self.logs_directory = Path(logging_cfg.get("logs_directory", "./logs"))
        self._log_run_path = None
        self.log_manager: Optional[ModuleLogManager] = None
        self.dashboard: Optional[LiveDashboard] = None
        if self.enable_live_logs:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_path = (self.logs_directory / f"{self.config.name}_{timestamp}").resolve()
            self._log_run_path = run_path
            self.log_manager = ModuleLogManager(
                run_directory=run_path,
                max_log_bytes=logging_cfg.get("max_log_file_bytes", 10 * 1024 * 1024),
                backup_count=logging_cfg.get("log_backup_count", 5),
            )
            dashboard_enabled = sys.stdout.isatty()
            self.dashboard = LiveDashboard(
                self.log_manager,
                enabled=dashboard_enabled,
            )
            self.dashboard.start()
            time.sleep(0.05)
            self.log_manager.register_module("orchestrator")
            # Begin orchestrator log capture early
            self._orchestrator_logger_ctx = self.log_manager.capture_logger("orchestrator")
            self._orchestrator_logger_ctx.__enter__()
        
        logger = get_logger("orchestrator", force_setup=True)
        logger.info(f"Initializing pipeline orchestrator: {self.config.name}")
        
        # Initialize components
        self.dependency_graph = DependencyGraph(self.config)
        logger.debug("Dependency graph initialized")
        
        # Resource manager
        max_cpus = self.config.resources.get("max_cpus")
        max_gpus = self.config.resources.get("max_gpus")
        self.resource_manager = ResourceManager(
            max_cpus=max_cpus,
            max_gpus=max_gpus
        )
        logger.info(
            f"Resource manager initialized: {self.resource_manager.total_cpus} CPUs, "
            f"{self.resource_manager.total_gpus} GPUs"
        )
        
        # Checkpoint manager
        checkpoint_config = self.config.checkpoint
        checkpoint_enabled = checkpoint_config.get("enabled", True)
        if checkpoint_enabled:
            checkpoint_dir = checkpoint_config.get("directory", "./.checkpoints")
            self.checkpoint_manager = PipelineCheckpointManager(
                checkpoint_directory=checkpoint_dir,
                enabled=True
            )
            logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
        else:
            self.checkpoint_manager = None
            logger.info("Checkpoint manager disabled")
        
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
            self.executor = SequentialExecutor(
                self.resource_manager,
                log_manager=self.log_manager
            )
            logger.info("Sequential executor initialized")
        else:
            self.executor = ParallelExecutor(
                resource_manager=self.resource_manager,
                worker_type=execution_config["worker_type"],
                failure_policy=execution_config["failure_policy"],
                max_nested_depth=execution_config.get("max_nested_depth"),
                ipc_manager=self.ipc_manager,
                log_manager=self.log_manager
            )
            logger.info(
                f"Parallel executor initialized: worker_type={execution_config['worker_type']}, "
                f"failure_policy={execution_config['failure_policy']}"
            )
        
        # Module cache
        self._modules: Dict[str, BaseModule] = {}
        
        # Execution state
        self._execution_results: Dict[str, Any] = {}
        
        # Queue system
        self.module_queue = ModuleQueue()
        self.queue_manager: Optional[ExecutionQueueManager] = None
    
    def load_modules(self):
        """Load all modules from configuration."""
        logger.info(f"Loading {len(self.config.modules)} modules")
        for module_config in self.config.modules:
            module_name = module_config["name"]
            if module_name not in self._modules:
                logger.debug(f"Loading module: {module_name}")
                module = ModuleLoader.load_module(module_config)
                self._modules[module_name] = module
                logger.debug(f"Module loaded: {module_name}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the pipeline using queue-based execution.
        
        Automatically re-executes failed modules from previous runs.
        All modules are queued and executed through a single execution point (queue worker).
        
        Returns:
            Dictionary mapping module names to results (or errors)
        """
        logger.info(f"Starting pipeline execution: {self.config.name}")
        try:
            # Check for failed modules from previous runs
            failed_modules = self.results_manager.get_failed_modules()
            if failed_modules:
                logger.info(
                    f"Found {len(failed_modules)} failed module(s) from previous run(s): {failed_modules}. "
                    f"These will be re-executed."
                )

            # Load modules
            self.load_modules()

            # Initialize queue manager
            self.queue_manager = ExecutionQueueManager(
                queue=self.module_queue,
                executor=self.executor,
                resource_manager=self.resource_manager,
                dependency_graph=self.dependency_graph,
                results_manager=self.results_manager,
                modules=self._modules,
                log_manager=self.log_manager
            )
            
            # Start queue worker thread
            self.queue_manager.start()
            logger.info("Queue-based execution started")

            # Track which modules have been queued
            queued_modules: set = set()
            
            # Main loop: continuously check for ready modules and enqueue them
            while True:
                # Get ready modules ordered by dependency completion time
                ready_modules = self.dependency_graph.get_ready_modules_with_dependency_order()
                
                # Enqueue ready modules that haven't been queued yet
                for module_name, dependency_completion_time in ready_modules:
                    if module_name in queued_modules:
                        continue
                    
                    if self.dependency_graph.is_completed(module_name):
                        continue
                    
                    # Get module config
                    module_config = self.config.get_module_config(module_name)
                    if module_config is None:
                        raise ConfigurationError(f"Module config not found: {module_name}")
                    
                    # Get resource requirements
                    resources = module_config.get("resources", {})
                    resource_reqs = {
                        "cpus": resources.get("cpus", 1),
                        "gpus": resources.get("gpus", 0)
                    }
                    
                    # Get module
                    module = self._modules[module_name]
                    
                    # Create context (without resource allocation - queue worker will handle it)
                    dependency_results = self.results_manager.get_dependency_results(module_name)
                    
                    # Create nested execution function for context
                    execute_tasks_fn = None
                    if isinstance(self.executor, ParallelExecutor) and self.executor.worker_type == "process":
                        # Leave as None for process workers - executor will set it via IPC
                        execute_tasks_fn = None
                    else:
                        # For sequential or thread workers, we can use nested functions
                        def create_execute_tasks_fn(module_name: str):
                            def execute_tasks(tasks):
                                return self._execute_nested_tasks(module_name, tasks)
                            return execute_tasks
                        execute_tasks_fn = create_execute_tasks_fn(module_name)
                    
                    context = ModuleContext(
                        module_name=module_name,
                        pipeline_name=self.config.name,
                        resources={
                            "cpus": resource_reqs["cpus"],
                            "gpus": resource_reqs["gpus"],
                            "gpu_ids": [],
                            "cuda_visible_devices": None,
                            "pytorch_devices": [],
                            "gpu_names": []
                        },
                        dependency_results=dependency_results,
                        execute_tasks_fn=execute_tasks_fn
                    )
                    
                    # Enqueue module
                    self.queue_manager.enqueue_module(
                        module_name=module_name,
                        module=module,
                        context=context,
                        resource_reqs=resource_reqs,
                        dependency_completion_time=dependency_completion_time
                    )
                    queued_modules.add(module_name)
                    logger.debug(f"Enqueued module '{module_name}'")
                
                # Check if all modules are completed
                if self.dependency_graph.all_completed():
                    logger.info("All modules completed")
                    break
                
                # Check if queue is idle (empty and no running modules)
                if self.queue_manager.is_idle() and len(queued_modules) == len(self._modules):
                    # All modules have been queued and queue is idle
                    # Wait a bit to see if any modules are still running
                    time.sleep(0.1)
                    if self.queue_manager.is_idle():
                        logger.info("Queue is idle and all modules have been queued")
                        break
                
                # Small sleep to avoid busy waiting
                time.sleep(0.05)
            
            # Wait for queue to finish processing
            logger.info("Waiting for queue worker to complete all execution...")
            max_wait_time = 300  # 5 minutes max wait
            wait_interval = 0.5
            waited = 0.0
            while not self.queue_manager.is_idle() and waited < max_wait_time:
                time.sleep(wait_interval)
                waited += wait_interval
            
            if not self.queue_manager.is_idle():
                logger.warning(f"Queue worker did not become idle within {max_wait_time} seconds")
            
            # Stop queue manager
            self.queue_manager.stop(timeout=10.0)
            
            # Collect results
            all_results = self.results_manager.get_all_results()
            
            # Also get failed module results
            for module_name in self._modules:
                if module_name not in all_results:
                    # Check if it failed
                    status = self.results_manager.get_module_status(module_name)
                    if status == BaseModule.ModuleStatus.FAILED:
                        # Try to get error from checkpoint
                        if self.checkpoint_manager:
                            try:
                                result, _ = self.checkpoint_manager.load_result(module_name)
                                if isinstance(result, Exception):
                                    all_results[module_name] = result
                            except Exception:
                                pass
            
            self._execution_results = all_results

            # Log final summary
            total_modules = len(all_results)
            successful_modules = sum(1 for r in all_results.values() if not isinstance(r, Exception))
            failed_modules = total_modules - successful_modules
            logger.info(
                f"Pipeline execution completed: {successful_modules}/{total_modules} modules succeeded"
            )
            if failed_modules > 0:
                logger.warning(f"{failed_modules} module(s) failed during execution")
                
                # Check fail-fast policy
                if self.config.execution["failure_policy"] == "fail_fast":
                    logger.error("Fail-fast policy: Pipeline execution completed with failures")

            return all_results
        finally:
            if self.enable_live_logs and self.log_manager:
                # Close early-entered orchestrator capture contexts
                if hasattr(self, "_orchestrator_logger_ctx"):
                    self._orchestrator_logger_ctx.__exit__(None, None, None)
                # if hasattr(self, "_orchestrator_stream_ctx"):
                    # self._orchestrator_stream_ctx.__exit__(None, None, None)
            if self.dashboard and self.enable_live_logs:
                # Restore original console logging handlers
                if hasattr(self, "_original_handlers"):
                    import logging
                    root = logging.getLogger()
                    for h in self._original_handlers:
                        root.addHandler(h)
                self.dashboard.stop()
            if self.log_manager and self.enable_live_logs:
                self.log_manager.shutdown()
    
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
        
        logger.debug(
            f"Module {requesting_module_name} requesting nested execution of {len(tasks)} task(s)"
        )
        
        # Validate all tasks are callable
        for i, task in enumerate(tasks):
            if not callable(task):
                from pipeline_orchestrator.exceptions import NestedExecutionError
                logger.error(
                    f"Module {requesting_module_name}: Task {i} is not callable: {task}"
                )
                return [NestedExecutionError(f"Task {i} is not callable: {task}")] * len(tasks)
        
        # Get execution configuration
        execution_config = self.config.execution
        worker_type = execution_config["worker_type"]
        max_nested_depth = execution_config.get("max_nested_depth")
        
        # Validate nested depth (if limited)
        # Note: We don't track depth currently, but could be added
        if max_nested_depth is not None and max_nested_depth <= 0:
            from pipeline_orchestrator.exceptions import NestedExecutionError
            logger.error(
                f"Module {requesting_module_name}: Maximum nested execution depth exceeded"
            )
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
            logger.debug(
                f"Executing {len(tasks)} nested task(s) for {requesting_module_name} "
                f"using parallel executor"
            )
            return self.executor._execute_nested_tasks(tasks, self.results_manager)
        else:
            # Sequential executor - execute tasks sequentially
            logger.debug(
                f"Executing {len(tasks)} nested task(s) for {requesting_module_name} "
                f"sequentially"
            )
            results = []
            for i, task in enumerate(tasks):
                try:
                    result = task()
                    results.append(result)
                except Exception as e:
                    logger.warning(
                        f"Nested task {i} for {requesting_module_name} failed: {e}"
                    )
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
    
    def restart(self) -> Dict[str, Any]:
        """
        Restart pipeline execution from failed modules.
        
        This method explicitly restarts the pipeline to re-execute failed modules.
        It's functionally equivalent to execute() since execute() automatically
        re-executes failed modules. This method is provided for clarity and
        explicit restart semantics.
        
        Returns:
            Dictionary mapping module names to results (or errors)
        """
        logger.info(f"Restarting pipeline execution: {self.config.name}")
        
        # Check for failed modules
        failed_modules = self.results_manager.get_failed_modules()
        if not failed_modules:
            logger.info("No failed modules found. Pipeline may be complete or fresh.")
        else:
            logger.info(
                f"Restarting with {len(failed_modules)} failed module(s): {failed_modules}"
            )
        
        # Execute pipeline (will automatically include failed modules)
        return self.execute()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up pipeline orchestrator resources")
        
        # Stop queue manager if running
        if self.queue_manager is not None:
            try:
                self.queue_manager.stop(timeout=5.0)
                logger.debug("Queue manager stopped")
            except Exception as e:
                logger.warning(f"Error stopping queue manager: {e}")
        
        if self.ipc_manager:
            self.ipc_manager.cleanup()
            logger.debug("IPC manager cleaned up")

