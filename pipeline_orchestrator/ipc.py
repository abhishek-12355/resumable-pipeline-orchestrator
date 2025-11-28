"""Inter-process communication utilities for nested task execution."""

import io
import multiprocessing
import queue
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline_orchestrator.exceptions import NestedExecutionError
from pipeline_orchestrator.logging import LogEvent


class WorkerIPCManager:
    """Manages IPC channels for worker processes."""
    
    def __init__(self):
        """Initialize IPC manager."""
        # Use a Manager to create queues that can be pickled and shared
        # Manager must be started before creating queues
        self._manager = multiprocessing.Manager()
        # Per-worker communication channels
        # {worker_id: (request_queue, response_queue)}
        self._channels: Dict[str, Tuple[multiprocessing.Queue, multiprocessing.Queue]] = {}
        self._lock = threading.Lock()
    
    def create_channel(self, worker_id: str) -> Tuple[multiprocessing.Queue, multiprocessing.Queue]:
        """
        Create request/response queue pair for a worker.
        
        Uses Manager queues so they can be pickled and shared across processes.
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Tuple of (request_queue, response_queue)
        """
        with self._lock:
            if worker_id in self._channels:
                raise NestedExecutionError(f"Channel already exists for worker '{worker_id}'")
            
            # Use Manager queues which can be pickled and shared
            request_queue = self._manager.Queue()
            response_queue = self._manager.Queue()
            
            self._channels[worker_id] = (request_queue, response_queue)
            
            return request_queue, response_queue
    
    def get_channel(self, worker_id: str) -> Optional[Tuple[multiprocessing.Queue, multiprocessing.Queue]]:
        """
        Get communication channel for a worker.
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Tuple of (request_queue, response_queue) or None if not found
        """
        with self._lock:
            return self._channels.get(worker_id)
    
    def remove_channel(self, worker_id: str):
        """
        Remove communication channel for a worker.
        
        Args:
            worker_id: Unique identifier for the worker
        """
        with self._lock:
            if worker_id in self._channels:
                request_queue, response_queue = self._channels[worker_id]
                
                # Close queues
                try:
                    request_queue.close()
                    response_queue.close()
                except Exception:
                    pass
                
                del self._channels[worker_id]
    
    def get_all_channels(self) -> Dict[str, Tuple[multiprocessing.Queue, multiprocessing.Queue]]:
        """
        Get all communication channels.
        
        Returns:
            Dictionary mapping worker_id to (request_queue, response_queue)
        """
        with self._lock:
            return self._channels.copy()
    
    def wait_for_requests(
        self,
        timeout: Optional[float] = None
    ) -> List[Tuple[str, Any]]:
        """
        Wait for requests from any worker channel (non-blocking check).
        
        Args:
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            List of tuples (worker_id, request_data) for available requests
        """
        requests = []
        
        with self._lock:
            channels = self._channels.copy()
        
        for worker_id, (request_queue, _) in channels.items():
            try:
                # Non-blocking get
                request_data = request_queue.get_nowait()
                requests.append((worker_id, request_data))
            except queue.Empty:
                # No request from this worker
                continue
        
        return requests
    
    def send_response(self, worker_id: str, response_data: Any):
        """
        Send response to a worker.
        
        Args:
            worker_id: Unique identifier for the worker
            response_data: Response data to send
        """
        with self._lock:
            if worker_id not in self._channels:
                raise NestedExecutionError(f"No channel found for worker '{worker_id}'")
            
            _, response_queue = self._channels[worker_id]
            
            try:
                response_queue.put(response_data)
            except Exception as e:
                raise NestedExecutionError(f"Failed to send response to worker '{worker_id}': {e}")
    
    def cleanup(self):
        """Cleanup all channels."""
        with self._lock:
            for worker_id in list(self._channels.keys()):
                self.remove_channel(worker_id)
        # Shutdown the manager
        try:
            self._manager.shutdown()
        except Exception:
            pass


class NestedTaskRequest:
    """Request for nested task execution."""
    
    def __init__(self, task_id: str, tasks: List[Any], worker_id: str):
        """
        Initialize nested task request.
        
        Args:
            task_id: Unique identifier for this request
            tasks: List of tasks to execute
            worker_id: ID of the requesting worker
        """
        self.task_id = task_id
        self.tasks = tasks
        self.worker_id = worker_id


class NestedTaskResponse:
    """Response from nested task execution."""
    
    def __init__(self, task_id: str, results: List[Any]):
        """
        Initialize nested task response.
        
        Args:
            task_id: Unique identifier for the request
            results: List of results/errors in same order as tasks
        """
        self.task_id = task_id
        self.results = results


class WorkerIPCClient:
    """Client for worker processes to communicate with orchestrator."""
    
    def __init__(
        self,
        worker_id: str,
        request_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue
    ):
        """
        Initialize IPC client for worker.
        
        Args:
            worker_id: Unique identifier for the worker
            request_queue: Queue for sending requests to orchestrator
            response_queue: Queue for receiving responses from orchestrator
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
    
    def execute_tasks(self, tasks: List[Any], timeout: Optional[float] = None) -> List[Any]:
        """
        Request orchestrator to execute tasks in parallel.
        
        Args:
            tasks: List of tasks to execute
            timeout: Timeout in seconds (None for no timeout)
            
        Returns:
            List of results/errors in same order as tasks
            
        Raises:
            NestedExecutionError: If execution fails or times out
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create request
        request = NestedTaskRequest(
            task_id=task_id,
            tasks=tasks,
            worker_id=self.worker_id
        )
        
        # Send request
        try:
            self.request_queue.put(request)
        except Exception as e:
            raise NestedExecutionError(f"Failed to send task request: {e}")
        
        # Wait for response
        try:
            if timeout is None:
                response = self.response_queue.get()
            else:
                response = self.response_queue.get(timeout=timeout)
        except queue.Empty:
            raise NestedExecutionError(f"Task execution timed out after {timeout} seconds")
        except Exception as e:
            raise NestedExecutionError(f"Failed to receive task response: {e}")
        
        # Validate response
        if not isinstance(response, NestedTaskResponse):
            raise NestedExecutionError(f"Invalid response type: {type(response)}")
        
        if response.task_id != task_id:
            raise NestedExecutionError(
                f"Response task_id mismatch: expected {task_id}, got {response.task_id}"
            )
        
        return response.results


class LogEventPipe:
    """Simple wrapper around a multiprocessing queue for log forwarding."""

    def __init__(self):
        self._queue: multiprocessing.Queue = multiprocessing.Queue()

    def sender(self) -> multiprocessing.Queue:
        """Queue endpoint for the worker process."""
        return self._queue

    def receiver(self) -> multiprocessing.Queue:
        """Queue endpoint for the orchestrator."""
        return self._queue

    def close(self):
        """Close the underlying queue."""
        try:
            self._queue.close()
        except Exception:
            pass


class _QueueStreamProxy(io.TextIOBase):
    """Redirects stdout/stderr writes into a multiprocessing queue."""

    def __init__(
        self,
        module_name: str,
        stream: str,
        queue_obj: multiprocessing.Queue,
        counter: Dict[str, int],
    ):
        self.module_name = module_name
        self.stream = stream
        self.queue = queue_obj
        self.counter = counter
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(data)

    def flush(self):
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def isatty(self):
        return False

    def _emit(self, message: str):
        seq = self.counter["value"]
        self.counter["value"] += 1
        event = LogEvent(
            timestamp=time.time(),
            module_name=self.module_name,
            stream=self.stream,
            message=message.rstrip("\r"),
            sequence=seq,
        )
        try:
            self.queue.put(event)
        except Exception:
            # Swallow errors to avoid crashing the worker
            pass


@contextmanager
def capture_process_logs(module_name: str, log_queue: Optional[multiprocessing.Queue]):
    """Context manager that redirects stdout/stderr to a queue for aggregation."""
    if log_queue is None:
        yield
        return
    counter = {"value": 0}
    original_stdout, original_stderr = sys.stdout, sys.stderr
    proxy_stdout = _QueueStreamProxy(module_name, "stdout", log_queue, counter)
    proxy_stderr = _QueueStreamProxy(module_name, "stderr", log_queue, counter)
    sys.stdout = proxy_stdout
    sys.stderr = proxy_stderr
    try:
        yield
    finally:
        proxy_stdout.flush()
        proxy_stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@contextmanager
def capture_process_logger(
    module_name: str,
    log_queue: Optional[multiprocessing.Queue],
    base_formatter: Optional["logging.Formatter"] = None,
):
    """
    Context manager that forwards Python logging.* from a worker process
    into the multiprocessing log queue, preserving the logger formatter.

    Unlike capture_process_logs() (stdout/stderr), this captures logging
    subsystem events such as logger.info(), logger.error(), etc.
    """

    if log_queue is None:
        yield
        return

    import logging

    from pipeline_orchestrator.logging import LogEvent

    class _QueueLoggingHandler(logging.Handler):
        def __init__(self, module_name: str, queue_obj: multiprocessing.Queue):
            super().__init__()
            self.module_name = module_name
            self.queue = queue_obj
            self.sequence = 0

        def emit(self, record: logging.LogRecord):
            # Preserve formatting if possible
            msg = self.format(record)
            event = LogEvent(
                timestamp=record.created,
                module_name=self.module_name,
                stream=(record.levelname or "LOG").lower(),
                message=msg,
                sequence=self.sequence,
            )
            self.sequence += 1
            try:
                self.queue.put(event)
            except Exception:
                pass

    root_logger = logging.getLogger()
    handler = _QueueLoggingHandler(module_name, log_queue)

    # Mirror formatter from existing handler if available
    if base_formatter is None and root_logger.handlers:
        first = root_logger.handlers[0]
        if first.formatter is not None:
            base_formatter = first.formatter

    if base_formatter is not None:
        handler.setFormatter(base_formatter)

    root_logger.addHandler(handler)

    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()

