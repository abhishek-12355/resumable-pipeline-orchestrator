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
            # logger.info(f"Loading pipeline configuration from: {config_path}")
            self.config = PipelineConfig.from_yaml_file(config_path)
        else:
            # logger.info("Using provided pipeline configuration")
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
            # self._orchestrator_stream_ctx = self.log_manager.capture_streams("orchestrator")
            self._orchestrator_logger_ctx = self.log_manager.capture_logger("orchestrator")
            # self._orchestrator_stream_ctx.__enter__()
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
        Execute the pipeline.
        
        Automatically re-executes failed modules from previous runs.
        Failed modules are identified from checkpoint metadata and included
        in execution batches.
        
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

            # Get execution order (batches of parallelizable modules)
            # Failed modules are not marked as completed, so they will be included in batches
            execution_batches = self.dependency_graph.get_execution_order()
            logger.info(f"Pipeline execution plan: {len(execution_batches)} batch(es)")

            # Execute batches sequentially, modules in batch in parallel
            all_results = {}

            for batch_idx, batch in enumerate(execution_batches, 1):
                logger.info(f"Executing batch {batch_idx}/{len(execution_batches)}: {batch}")
                # Skip already completed modules
                batch = [m for m in batch if not self.dependency_graph.is_completed(m)]
                if not batch:
                    logger.debug(f"Batch {batch_idx}: All modules already completed, skipping")
                    continue

                logger.info(f"Batch {batch_idx}: Executing {len(batch)} module(s)")

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
                        logger.debug(
                            f"Reserved resources for {module_name}: {cpus} CPUs, "
                            f"{len(assigned_gpus)} GPUs {assigned_gpus if assigned_gpus else ''}"
                        )
                    except ResourceError as e:
                        logger.error(f"Failed to reserve resources for {module_name}: {e}")
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

                    if self.log_manager:
                        self.log_manager.register_module(module_name)

                    # Create context
                    dependency_results = self.results_manager.get_dependency_results(module_name)

                    # Create nested execution function for context
                    # Note: For process workers, we don't set execute_tasks_fn here because
                    # nested functions can't be pickled. The executor will set it in the
                    # worker process using the IPC client.
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
                            "cpus": cpus,
                            "gpus": gpus,
                            "gpu_ids": assigned_gpus,
                            "cuda_visible_devices": cuda_visible,
                            "pytorch_devices": pytorch_devices,
                            "gpu_names": gpu_names
                        },
                        dependency_results=dependency_results,
                        execute_tasks_fn=execute_tasks_fn
                    )

                    batch_modules[module_name] = module
                    batch_contexts[module_name] = context

                # Execute batch
                logger.info(f"Batch {batch_idx}: Starting execution of {len(batch_modules)} module(s)")
                batch_results = self.executor.execute_modules(
                    batch_modules,
                    batch_contexts,
                    self.results_manager
                )

                # Release resources
                for module_name in batch:
                    self.resource_manager.release_resources(module_name)
                    logger.debug(f"Released resources for {module_name}")

                # Collect results
                all_results.update(batch_results)

                # Log batch completion
                successful = sum(1 for r in batch_results.values() if not isinstance(r, Exception))
                failed = len(batch_results) - successful
                logger.info(
                    f"Batch {batch_idx} completed: {successful} succeeded, {failed} failed"
                )

                # Check for failures in fail-fast mode
                if self.config.execution["failure_policy"] == "fail_fast":
                    for module_name, result in batch_results.items():
                        if isinstance(result, Exception):
                            logger.error(
                                f"Module {module_name} failed in fail-fast mode. "
                                f"Stopping pipeline execution."
                            )
                            # Stop execution on first failure
                            return all_results

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
        if self.ipc_manager:
            self.ipc_manager.cleanup()
            logger.debug("IPC manager cleaned up")

