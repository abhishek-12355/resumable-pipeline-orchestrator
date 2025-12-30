"""Queue management for module execution with dependency-aware ordering."""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from collections import deque

from pipeline_orchestrator.logging_config import get_logger
from pipeline_orchestrator.module import BaseModule
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.resources import ResourceManager
from pipeline_orchestrator.dependency import DependencyGraph
from pipeline_orchestrator.executor import BaseExecutor
from pipeline_orchestrator.results import ResultsManager

logger = get_logger(__name__)


@dataclass
class QueuedModule:
    """Represents a module waiting in the queue."""
    module_name: str
    resource_reqs: Dict[str, int]  # {"cpus": int, "gpus": int}
    queued_time: float
    dependency_completion_time: float  # When last dependency completed


class ModuleQueue:
    """Manages queued modules with dependency-aware ordering."""
    
    def __init__(self):
        """Initialize the module queue."""
        self._lock = threading.RLock()
        self._queue: List[QueuedModule] = []
        self._queued_modules: set = set()  # Track which modules are queued
    
    def enqueue(
        self,
        module_name: str,
        resource_reqs: Dict[str, int],
        dependency_completion_time: Optional[float] = None
    ):
        """
        Add a module to the queue.
        
        Args:
            module_name: Name of the module
            resource_reqs: Dictionary with "cpus" and "gpus" keys
            dependency_completion_time: Timestamp when dependencies completed (defaults to now)
        """
        with self._lock:
            if module_name in self._queued_modules:
                logger.debug(f"Module '{module_name}' already in queue, skipping")
                return
            
            queued_time = time.time()
            if dependency_completion_time is None:
                dependency_completion_time = queued_time
            
            queued_module = QueuedModule(
                module_name=module_name,
                resource_reqs=resource_reqs,
                queued_time=queued_time,
                dependency_completion_time=dependency_completion_time
            )
            
            # Insert in sorted order: by dependency_completion_time, then queued_time
            inserted = False
            for i, existing in enumerate(self._queue):
                if (queued_module.dependency_completion_time < existing.dependency_completion_time or
                    (queued_module.dependency_completion_time == existing.dependency_completion_time and
                     queued_module.queued_time < existing.queued_time)):
                    self._queue.insert(i, queued_module)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(queued_module)
            
            self._queued_modules.add(module_name)
            logger.debug(f"Enqueued module '{module_name}' (queue size: {len(self._queue)})")
    
    def dequeue(self, can_run: Optional[Callable[[str], bool]] = None) -> Optional[QueuedModule]:
        """
        Get the next module that can run.
        
        Args:
            can_run: Optional function that takes module_name and returns True if module can run
        
        Returns:
            QueuedModule if found, None otherwise
        """
        with self._lock:
            if not self._queue:
                return None
            
            if can_run is None:
                # No filter, return first module
                queued_module = self._queue.pop(0)
                self._queued_modules.discard(queued_module.module_name)
                return queued_module
            
            # Find first module that can run
            for i, queued_module in enumerate(self._queue):
                if can_run(queued_module.module_name):
                    self._queue.pop(i)
                    self._queued_modules.discard(queued_module.module_name)
                    logger.debug(f"Dequeued module '{queued_module.module_name}'")
                    return queued_module
            
            return None
    
    def remove(self, module_name: str) -> bool:
        """
        Remove a module from the queue.
        
        Args:
            module_name: Name of the module to remove
        
        Returns:
            True if module was removed, False if not found
        """
        with self._lock:
            if module_name not in self._queued_modules:
                return False
            
            self._queue = [m for m in self._queue if m.module_name != module_name]
            self._queued_modules.discard(module_name)
            logger.debug(f"Removed module '{module_name}' from queue")
            return True
    
    def is_queued(self, module_name: str) -> bool:
        """
        Check if a module is queued.
        
        Args:
            module_name: Name of the module
        
        Returns:
            True if module is queued, False otherwise
        """
        with self._lock:
            return module_name in self._queued_modules
    
    def get_queue_size(self) -> int:
        """
        Get number of queued modules.
        
        Returns:
            Number of modules in queue
        """
        with self._lock:
            return len(self._queue)
    
    def get_queue(self) -> List[QueuedModule]:
        """
        Get all queued modules (for monitoring).
        
        Returns:
            List of QueuedModule instances (copy)
        """
        with self._lock:
            return self._queue.copy()
    
    def clear(self):
        """Clear all modules from the queue."""
        with self._lock:
            self._queue.clear()
            self._queued_modules.clear()
            logger.debug("Queue cleared")


class ExecutionQueueManager:
    """Background worker that processes the execution queue - SINGLE EXECUTION POINT."""
    
    def __init__(
        self,
        queue: ModuleQueue,
        executor: BaseExecutor,
        resource_manager: ResourceManager,
        dependency_graph: DependencyGraph,
        results_manager: ResultsManager,
        modules: Dict[str, BaseModule],
        log_manager: Optional[Any] = None
    ):
        """
        Initialize execution queue manager.
        
        Args:
            queue: ModuleQueue instance
            executor: Executor instance for executing modules
            resource_manager: ResourceManager instance
            dependency_graph: DependencyGraph instance
            results_manager: ResultsManager instance
            modules: Dictionary mapping module names to module instances
            log_manager: Optional log manager
        """
        self.queue = queue
        self.executor = executor
        self.resource_manager = resource_manager
        self.dependency_graph = dependency_graph
        self.results_manager = results_manager
        self.modules = modules
        self.log_manager = log_manager
        
        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running_modules: Dict[str, threading.Thread] = {}  # module_name -> thread
        self._module_contexts: Dict[str, ModuleContext] = {}  # module_name -> context
        self._module_resource_reqs: Dict[str, Dict[str, int]] = {}  # module_name -> resource_reqs
    
    def enqueue_module(
        self,
        module_name: str,
        module: BaseModule,
        context: ModuleContext,
        resource_reqs: Dict[str, int],
        dependency_completion_time: Optional[float] = None
    ):
        """
        Add a module to the queue for execution.
        
        Args:
            module_name: Name of the module
            module: Module instance
            context: Execution context
            resource_reqs: Dictionary with "cpus" and "gpus" keys
            dependency_completion_time: Timestamp when dependencies completed
        """
        with self._lock:
            # Store module and context for later execution
            self.modules[module_name] = module
            self._module_contexts[module_name] = context
            self._module_resource_reqs[module_name] = resource_reqs
            
            # Add to queue
            self.queue.enqueue(module_name, resource_reqs, dependency_completion_time)
            logger.debug(f"Queued module '{module_name}' for execution")
    
    def start(self):
        """Start the background worker thread."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.warning("Queue worker thread already running")
                return
            
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="ExecutionQueueWorker",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Execution queue worker thread started")
    
    def stop(self, timeout: Optional[float] = None):
        """
        Stop the background worker thread.
        
        Args:
            timeout: Maximum time to wait for worker to stop (None for no timeout)
        """
        logger.info("Stopping execution queue worker...")
        self._stop_event.set()
        
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                logger.warning("Queue worker thread did not stop within timeout")
            else:
                logger.info("Execution queue worker thread stopped")
        
        # Wait for running modules to complete
        logger.info(f"Waiting for {len(self._running_modules)} running module(s) to complete...")
        for module_name, thread in list(self._running_modules.items()):
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Module '{module_name}' did not complete within timeout")
    
    def _can_execute(self, module_name: str) -> bool:
        """
        Check if a module can execute (dependencies satisfied AND resources available).
        
        Args:
            module_name: Name of the module
        
        Returns:
            True if module can execute, False otherwise
        """
        # Check dependencies are satisfied
        if module_name not in self.modules:
            return False
        
        deps = self.dependency_graph.get_dependencies(module_name)
        if not all(self.dependency_graph.is_completed(dep) for dep in deps):
            return False
        
        # Check resources are available
        resource_reqs = self._module_resource_reqs.get(module_name)
        if not resource_reqs:
            return False
        
        cpus = resource_reqs.get("cpus", 1)
        gpus = resource_reqs.get("gpus", 0)
        
        return self.resource_manager.validate_resources(cpus, gpus)
    
    def _worker_loop(self):
        """Main worker loop that processes the queue."""
        logger.info("Queue worker loop started")
        
        while not self._stop_event.is_set():
            try:
                # Check if we can execute any module from queue
                queued_module = self.queue.dequeue(can_run=self._can_execute)
                
                if queued_module is None:
                    # No modules can run, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Execute module in a separate thread (to allow parallel execution)
                module_name = queued_module.module_name
                thread = threading.Thread(
                    target=self._execute_module,
                    args=(module_name,),
                    name=f"ModuleExecutor-{module_name}",
                    daemon=True
                )
                
                with self._lock:
                    self._running_modules[module_name] = thread
                
                thread.start()
                
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("Queue worker loop stopped")
    
    def _execute_module(self, module_name: str):
        """
        Execute a single module.
        
        Args:
            module_name: Name of the module to execute
        """
        try:
            logger.info(f"Starting execution of module '{module_name}'")
            
            # Get module and context
            module = self.modules.get(module_name)
            context = self._module_contexts.get(module_name)
            resource_reqs = self._module_resource_reqs.get(module_name)
            
            if module is None or context is None or resource_reqs is None:
                logger.error(f"Module '{module_name}' not found in stored modules/contexts")
                return
            
            # Register module for logging
            if self.log_manager:
                self.log_manager.register_module(module_name)
            
            # Set status to PENDING
            from pipeline_orchestrator.module import BaseModule
            self.results_manager.save_result(
                module_name, None, is_error=False,
                status=BaseModule.ModuleStatus.PENDING
            )
            
            # Try to reserve resources
            cpus = resource_reqs.get("cpus", 1)
            gpus = resource_reqs.get("gpus", 0)
            
            success, assigned_gpus, cuda_visible = self.resource_manager.try_reserve_resources(
                module_name, cpus, gpus
            )
            
            if not success:
                logger.warning(f"Failed to reserve resources for '{module_name}', re-queuing")
                # Re-queue the module (preserve dependency completion time)
                dependency_completion_time = self.dependency_graph.get_dependency_completion_time(module_name)
                self.queue.enqueue(module_name, resource_reqs, dependency_completion_time)
                return
            
            # Update context with GPU information
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
            
            context.resources.update({
                "gpu_ids": assigned_gpus,
                "cuda_visible_devices": cuda_visible,
                "pytorch_devices": pytorch_devices,
                "gpu_names": gpu_names
            })
            
            # Set status to IN_PROGRESS
            self.results_manager.save_result(
                module_name, None, is_error=False,
                status=BaseModule.ModuleStatus.IN_PROGRESS
            )
            
            # Execute module
            try:
                result = self.executor.execute_module(module_name, module, context)
                
                # Set status to SUCCESS
                self.results_manager.save_result(
                    module_name, result, is_error=False,
                    status=BaseModule.ModuleStatus.SUCCESS
                )
                logger.info(f"Module '{module_name}' completed successfully")
                
            except Exception as e:
                logger.error(f"Module '{module_name}' execution failed: {e}", exc_info=True)
                
                # Set status to FAILED
                self.results_manager.save_result(
                    module_name, e, is_error=True,
                    status=BaseModule.ModuleStatus.FAILED
                )
            
            finally:
                # Release resources
                self.resource_manager.release_resources(module_name)
                logger.debug(f"Released resources for '{module_name}'")
                
                # Unregister module from logging
                if self.log_manager:
                    self.log_manager.unregister_module(module_name)
        
        finally:
            # Remove from running modules
            with self._lock:
                self._running_modules.pop(module_name, None)
    
    def get_running_count(self) -> int:
        """
        Get number of currently running modules.
        
        Returns:
            Number of modules currently executing
        """
        with self._lock:
            return len(self._running_modules)
    
    def is_idle(self) -> bool:
        """
        Check if queue is empty and no modules running.
        
        Returns:
            True if queue is empty and no modules running, False otherwise
        """
        with self._lock:
            return self.queue.get_queue_size() == 0 and len(self._running_modules) == 0

