"""
Global routing context for multiprocess logging.
Used by QueueLoggingHandler to know which module/queue should receive logs.
"""

_current_module_name = None
_current_log_queue = None


def set_logging_context(module_name, queue):
    """Activate routing for the currently executing module inside this worker."""
    global _current_module_name, _current_log_queue
    prev_name = _current_module_name
    prev_queue = _current_log_queue
    _current_module_name = module_name
    _current_log_queue = queue
    return prev_name, prev_queue


def reset_logging_context(prev_name, prev_queue):
    """Restore previous logging routing context after nested execution."""
    global _current_module_name, _current_log_queue
    _current_module_name = prev_name
    _current_log_queue = prev_queue


def get_logging_context():
    """Return (module_name, queue) for currently active routing."""
    return _current_module_name, _current_log_queue