"""Execution engines for pipeline orchestrator."""

import os
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future

from pipeline_orchestrator.exceptions import ModuleExecutionError, ResourceError, NestedExecutionError
from pipeline_orchestrator.resources import ResourceManager
from pipeline_orchestrator.ipc import WorkerIPCManager, NestedTaskRequest, NestedTaskResponse
from pipeline_orchestrator.module import BaseModule
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.results import ResultsManager


class BaseExecutor(ABC):
    """Base class for execution engines."""
    
    @abstractmethod
    def execute_module(
        self,
        module_name: str,
        module: BaseModule,
        context: ModuleContext
    ) -> Any:
        """
        Execute a module.
        
        Args:
            module_name: Name of the module
            module: Module instance to execute
            context: Execution context
            
        Returns:
            Module result
        """
        pass
    
    @abstractmethod
    def execute_modules(
        self,
        modules: Dict[str, BaseModule],
        contexts: Dict[str, ModuleContext],
        results_manager: ResultsManager
    ) -> Dict[str, Any]:
        """
        Execute multiple modules.
        
        Args:
            modules: Dictionary mapping module names to module instances
            contexts: Dictionary mapping module names to execution contexts
            results_manager: Results manager for saving results
            
        Returns:
            Dictionary mapping module names to results (or errors)
        """
        pass


class SequentialExecutor(BaseExecutor):
    """Executor for sequential module execution."""
    
    def __init__(self, resource_manager: ResourceManager):
        """
        Initialize sequential executor.
        
        Args:
            resource_manager: Resource manager instance
        """
        self.resource_manager = resource_manager
    
    def execute_module(
        self,
        module_name: str,
        module: BaseModule,
        context: ModuleContext
    ) -> Any:
        """Execute a module sequentially."""
        try:
            result = module.run(context)
            return result
        except Exception as e:
            raise ModuleExecutionError(f"Module '{module_name}' execution failed: {e}") from e
    
    def execute_modules(
        self,
        modules: Dict[str, BaseModule],
        contexts: Dict[str, ModuleContext],
        results_manager: ResultsManager
    ) -> Dict[str, Any]:
        """Execute modules sequentially."""
        results = {}
        
        for module_name, module in modules.items():
            context = contexts[module_name]
            
            try:
                result = self.execute_module(module_name, module, context)
                results[module_name] = result
                results_manager.save_result(module_name, result)
            except Exception as e:
                results[module_name] = e
        
        return results


class _ModuleWorker:
    """Worker function for module execution in parallel mode."""
    
    @staticmethod
    def thread_worker(
        module_name: str,
        module: BaseModule,
        context: ModuleContext
    ) -> Tuple[str, Any]:
        """Thread worker function."""
        try:
            result = module.run(context)
            return (module_name, result)
        except Exception as e:
            return (module_name, ModuleExecutionError(f"Module '{module_name}' execution failed: {e}"))
    
    @staticmethod
    def process_worker(
        module_name: str,
        module: BaseModule,
        context: ModuleContext,
        worker_id: str,
        ipc_client: Optional[Any]
    ) -> Tuple[str, Any]:
        """Process worker function."""
        # Set CUDA_VISIBLE_DEVICES if GPUs allocated
        if context.resources.get("cuda_visible_devices"):
            os.environ["CUDA_VISIBLE_DEVICES"] = context.resources["cuda_visible_devices"]
        
        try:
            # Update context to use IPC client for nested execution
            if ipc_client:
                context._execute_tasks_fn = lambda tasks: ipc_client.execute_tasks(tasks)
            
            result = module.run(context)
            return (module_name, result)
        except Exception as e:
            return (module_name, ModuleExecutionError(f"Module '{module_name}' execution failed: {e}"))


class ParallelExecutor(BaseExecutor):
    """Executor for parallel module execution."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        worker_type: str = "process",
        failure_policy: str = "fail_fast",
        max_nested_depth: Optional[int] = None,
        ipc_manager: Optional[WorkerIPCManager] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            resource_manager: Resource manager instance
            worker_type: "thread" or "process"
            failure_policy: "fail_fast" or "collect_all"
            max_nested_depth: Maximum depth for nested execution (None for unlimited)
            ipc_manager: IPC manager for nested execution (process mode only)
        """
        self.resource_manager = resource_manager
        self.worker_type = worker_type
        self.failure_policy = failure_policy
        self.max_nested_depth = max_nested_depth
        self.ipc_manager = ipc_manager
        
        if worker_type == "process" and ipc_manager is None:
            self.ipc_manager = WorkerIPCManager()
    
    def execute_module(
        self,
        module_name: str,
        module: BaseModule,
        context: ModuleContext
    ) -> Any:
        """Execute a module (used by nested executor)."""
        try:
            result = module.run(context)
            return result
        except Exception as e:
            raise ModuleExecutionError(f"Module '{module_name}' execution failed: {e}") from e
    
    def execute_modules(
        self,
        modules: Dict[str, BaseModule],
        contexts: Dict[str, ModuleContext],
        results_manager: ResultsManager
    ) -> Dict[str, Any]:
        """Execute modules in parallel."""
        if self.worker_type == "thread":
            return self._execute_with_threads(modules, contexts, results_manager)
        else:
            return self._execute_with_processes(modules, contexts, results_manager)
    
    def _execute_with_threads(
        self,
        modules: Dict[str, BaseModule],
        contexts: Dict[str, ModuleContext],
        results_manager: ResultsManager
    ) -> Dict[str, Any]:
        """Execute modules using threads."""
        results = {}
        futures: Dict[Future, str] = {}
        
        with ThreadPoolExecutor(max_workers=len(modules)) as executor:
            # Submit all modules
            for module_name, module in modules.items():
                context = contexts[module_name]
                future = executor.submit(
                    _ModuleWorker.thread_worker,
                    module_name,
                    module,
                    context
                )
                futures[future] = module_name
            
            # Collect results as they complete
            for future in futures:
                module_name = futures[future]
                try:
                    _, result = future.result()
                    
                    if isinstance(result, Exception):
                        results[module_name] = result
                        if self.failure_policy == "fail_fast":
                            # Cancel remaining futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
                    else:
                        results[module_name] = result
                        results_manager.save_result(module_name, result)
                except Exception as e:
                    results[module_name] = e
                    if self.failure_policy == "fail_fast":
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
        
        return results
    
    def _execute_with_processes(
        self,
        modules: Dict[str, BaseModule],
        contexts: Dict[str, ModuleContext],
        results_manager: ResultsManager
    ) -> Dict[str, Any]:
        """Execute modules using processes."""
        results = {}
        futures: Dict[Future, str] = {}
        
        # Create IPC clients for each worker
        worker_clients = {}
        for module_name in modules:
            worker_id = f"worker_{module_name}_{id(modules[module_name])}"
            request_queue, response_queue = self.ipc_manager.create_channel(worker_id)
            
            from pipeline_orchestrator.ipc import WorkerIPCClient
            ipc_client = WorkerIPCClient(worker_id, request_queue, response_queue)
            worker_clients[module_name] = (worker_id, ipc_client)
        
        # Start nested execution handler in background thread
        nested_handler = threading.Thread(
            target=self._handle_nested_execution,
            args=(results_manager,),
            daemon=True
        )
        nested_handler.start()
        
        try:
            with ProcessPoolExecutor(max_workers=len(modules)) as executor:
                # Submit all modules
                for module_name, module in modules.items():
                    context = contexts[module_name]
                    worker_id, ipc_client = worker_clients[module_name]
                    
                    # Note: Modules and contexts need to be picklable for multiprocessing
                    # This is a limitation - modules must be importable
                    future = executor.submit(
                        _ModuleWorker.process_worker,
                        module_name,
                        module,
                        context,
                        worker_id,
                        ipc_client
                    )
                    futures[future] = module_name
                
                # Collect results as they complete
                for future in futures:
                    module_name = futures[future]
                    try:
                        _, result = future.result()
                        
                        if isinstance(result, Exception):
                            results[module_name] = result
                            if self.failure_policy == "fail_fast":
                                # Cancel remaining futures
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break
                        else:
                            results[module_name] = result
                            results_manager.save_result(module_name, result)
                    except Exception as e:
                        results[module_name] = e
                        if self.failure_policy == "fail_fast":
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break
        finally:
            # Cleanup IPC channels
            for module_name, (worker_id, _) in worker_clients.items():
                self.ipc_manager.remove_channel(worker_id)
        
        return results
    
    def _handle_nested_execution(self, results_manager: ResultsManager):
        """Handle nested execution requests from worker processes."""
        while True:
            # Check for nested execution requests
            requests = self.ipc_manager.wait_for_requests(timeout=0.1)
            
            for worker_id, request in requests:
                if not isinstance(request, NestedTaskRequest):
                    continue
                
                try:
                    # Execute nested tasks
                    # Note: This is a simplified version
                    # Full implementation would handle task execution with resource management
                    results = self._execute_nested_tasks(request.tasks, results_manager)
                    
                    # Send response
                    response = NestedTaskResponse(request.task_id, results)
                    self.ipc_manager.send_response(worker_id, response)
                except Exception as e:
                    # Send error response
                    error_response = NestedTaskResponse(
                        request.task_id,
                        [NestedExecutionError(f"Nested execution failed: {e}")] * len(request.tasks)
                    )
                    self.ipc_manager.send_response(worker_id, error_response)
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def _execute_nested_tasks(
        self,
        tasks: List[Any],
        results_manager: ResultsManager
    ) -> List[Any]:
        """
        Execute nested tasks.
        
        This is a placeholder - full implementation would:
        1. Validate resources for each task
        2. Execute tasks in parallel (threads/processes)
        3. Collect results/errors
        4. Return results in same order as tasks
        
        Args:
            tasks: List of tasks to execute
            results_manager: Results manager
            
        Returns:
            List of results/errors
        """
        # Simplified implementation - tasks are just callables
        results = []
        
        for task in tasks:
            try:
                if callable(task):
                    result = task()
                    results.append(result)
                else:
                    results.append(NestedExecutionError(f"Task is not callable: {task}"))
            except Exception as e:
                results.append(e)
        
        return results

