"""Live logging manager for per-module log streams."""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple
from contextlib import contextmanager


@dataclass(frozen=True)
class LogEvent:
    """Structured log event emitted by a module."""

    timestamp: float
    module_name: str
    stream: str
    message: str
    sequence: int

    def formatted_timestamp(self) -> str:
        """Return a human-friendly timestamp (HH:MM:SS.mmm)."""
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp)) + f".{int((self.timestamp % 1)*1000):03d}"


class _RollingFile:
    """Simple rotating file writer."""

    def __init__(self, path: Path, max_bytes: int, backup_count: int):
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        self._stream = self.path.open("a", encoding="utf-8")

    def write_line(self, line: str):
        """Write a line and rotate if necessary."""
        data = line if line.endswith("\n") else f"{line}\n"
        with self._lock:
            self._stream.write(data)
            self._stream.flush()
            if self.max_bytes and self._stream.tell() >= self.max_bytes:
                self._rotate()

    def close(self):
        """Close the stream."""
        with self._lock:
            try:
                self._stream.close()
            except Exception:
                pass

    def _rotate(self):
        """Rotate log files."""
        self._stream.close()
        if self.backup_count > 0:
            for index in range(self.backup_count - 1, 0, -1):
                src = self._backup_path(index)
                dest = self._backup_path(index + 1)
                if src.exists():
                    src.replace(dest)
            first_backup = self._backup_path(1)
            if self.path.exists():
                self.path.replace(first_backup)
        else:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._stream = self.path.open("a", encoding="utf-8")

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")


class _StreamProxy(io.TextIOBase):
    """File-like object that forwards writes to ModuleLogManager."""

    def __init__(self, manager: "ModuleLogManager", module_name: str, stream: str):
        self.manager = manager
        self.module_name = module_name
        self.stream = stream
        self._buffer = ""
        self._is_proxy = True

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.manager.log_text(self.module_name, line, stream=self.stream)
            else:
                self.manager.log_text(self.module_name, "", stream=self.stream)
        return len(data)

    def flush(self):
        if self._buffer:
            self.manager.log_text(self.module_name, self._buffer, stream=self.stream)
            self._buffer = ""

    def isatty(self):
        return False

class _LoggerToModuleHandler(logging.Handler):
    """Logging handler that forwards formatted records to ModuleLogManager."""

    def __init__(self, manager: "ModuleLogManager", module_name: str):
        super().__init__()
        self.manager = manager
        self.module_name = module_name

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            # Fallback to basic message if formatter fails
            msg = record.getMessage()
        stream = (record.levelname or "LOG").lower()
        # Use the record's creation time so ordering vs stdout is consistent
        self.manager.log_text(
            self.module_name,
            msg,
            stream=stream,
            timestamp=record.created,
        )


class ModuleLogManager:
    """Coordinates per-module log streams for UI/File sinks."""

    def __init__(
        self,
        run_directory: Path,
        history_size: int = 400,
        max_log_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.run_directory = run_directory
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.history_size = history_size
        self.max_log_bytes = max_log_bytes
        self.backup_count = max(1, backup_count)
        self._lock = threading.Lock()
        self._files: Dict[str, _RollingFile] = {}
        self._history: Dict[str, Deque[LogEvent]] = defaultdict(lambda: deque(maxlen=self.history_size))
        self._subscribers: Dict[str, "queue.SimpleQueue[LogEvent]"] = {}
        self._seq_counters: Dict[str, int] = defaultdict(int)

    def register_module(self, module_name: str):
        """Ensure we track log history/file for a module."""
        with self._lock:
            if module_name not in self._files:
                log_path = self.run_directory / f"{module_name}.log"
                self._files[module_name] = _RollingFile(
                    log_path,
                    max_bytes=self.max_log_bytes,
                    backup_count=self.backup_count,
                )

    def unregister_module(self, module_name: str):
        """Cleanup module resources."""
        with self._lock:
            file_writer = self._files.pop(module_name, None)
            self._history.pop(module_name, None)
            self._seq_counters.pop(module_name, None)
        if file_writer:
            file_writer.close()

    def subscribe(self, replay_history: bool = True) -> Tuple[str, "queue.SimpleQueue[LogEvent]"]:
        """Subscribe to log events, optionally replaying history."""
        import queue

        subscriber_id = str(uuid.uuid4())
        q: "queue.SimpleQueue[LogEvent]" = queue.SimpleQueue()
        with self._lock:
            self._subscribers[subscriber_id] = q
            if replay_history:
                history_snapshot = [event for events in self._history.values() for event in events]
        if replay_history:
            for event in history_snapshot:
                q.put(event)
        return subscriber_id, q

    def unsubscribe(self, subscriber_id: str):
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def log_text(
        self,
        module_name: str,
        message: str,
        stream: str = "log",
        timestamp: Optional[float] = None,
        sequence: Optional[int] = None,
    ):
        """Append raw text for a module."""
        message = message.rstrip("\r")
        event = self._create_event(
            module_name,
            stream,
            message,
            timestamp_override=timestamp,
            sequence_override=sequence,
        )
        self.ingest_event(event)

    def ingest_event(self, event: LogEvent):
        """Persist and broadcast an already-constructed event."""
        with self._lock:
            current = self._seq_counters[event.module_name]
            if event.sequence >= current:
                self._seq_counters[event.module_name] = event.sequence + 1
            history = self._history[event.module_name]
            history.append(event)
            file_writer = self._files.get(event.module_name)
        if file_writer:
            file_writer.write_line(
                f"{event.formatted_timestamp()} [{event.stream.upper()}] {event.message}"
            )
        self._broadcast(event)

    @contextmanager
    def capture_streams(self, module_name: str):
        """Redirect stdout/stderr for a module."""
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        proxy_stdout = _StreamProxy(self, module_name, "stdout")
        proxy_stderr = _StreamProxy(self, module_name, "stderr")
        sys.stdout = proxy_stdout
        sys.stderr = proxy_stderr
        try:
            yield
        finally:
            proxy_stdout.flush()
            proxy_stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    @contextmanager
    def capture_logger(self, module_name: str):
        """Attach a logging.Handler that forwards logger records to this module.

        This preserves the formatter configuration of the first existing
        handler on the root logger (if any), so log lines in the dashboard
        match console/file formatting as closely as possible.
        """
        root_logger = logging.getLogger()
        handler = _LoggerToModuleHandler(self, module_name)

        # Try to mirror the formatter of the first existing handler, if present
        if root_logger.handlers:
            base_handler = root_logger.handlers[0]
            if base_handler.formatter is not None:
                handler.setFormatter(base_handler.formatter)

        root_logger.addHandler(handler)
        try:
            yield
        finally:
            root_logger.removeHandler(handler)
            handler.close()

    def _create_event(
        self,
        module_name: str,
        stream: str,
        message: str,
        timestamp_override: Optional[float] = None,
        sequence_override: Optional[int] = None,
    ) -> LogEvent:
        with self._lock:
            if sequence_override is None:
                seq = self._seq_counters[module_name]
                self._seq_counters[module_name] += 1
            else:
                seq = sequence_override
                self._seq_counters[module_name] = max(
                    self._seq_counters[module_name], seq + 1
                )
        return LogEvent(
            timestamp=timestamp_override if timestamp_override is not None else time.time(),
            module_name=module_name,
            stream=stream,
            message=message,
            sequence=seq,
        )

    def _broadcast(self, event: LogEvent):
        with self._lock:
            subscribers = list(self._subscribers.values())
        for subscriber in subscribers:
            subscriber.put(event)

    def shutdown(self):
        """Flush and close all resources."""
        with self._lock:
            writers = list(self._files.values())
            self._files.clear()
        for writer in writers:
            writer.close()

