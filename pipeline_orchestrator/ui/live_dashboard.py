"""Rich-powered dashboard that renders per-module log panes."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from rich.columns import Columns
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pipeline_orchestrator.logging import LogEvent, ModuleLogManager


class LiveDashboard:
    """Continuously renders module logs via Rich."""

    def __init__(
        self,
        log_manager: ModuleLogManager,
        refresh_rate: float = 0.2,
        max_lines_per_module: int = 200,
        enabled: bool = True,
    ):
        self.log_manager = log_manager
        self.refresh_rate = max(0.05, refresh_rate)
        self.max_lines = max_lines_per_module
        self.console = Console()
        self.enabled = enabled and self.console.is_terminal
        self._subscriber_id: Optional[str] = None
        self._queue: Optional["queue.SimpleQueue[LogEvent]"] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffers: Dict[str, Deque[LogEvent]] = defaultdict(
            lambda: deque(maxlen=self.max_lines)
        )
        self._module_order: Dict[str, float] = {}

    def start(self):
        """Begin rendering in the background."""
        if not self.enabled or self._thread:
            return
        self._stop_event.clear()
        self._subscriber_id, self._queue = self.log_manager.subscribe(replay_history=True)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop rendering."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        if self._subscriber_id:
            self.log_manager.unsubscribe(self._subscriber_id)
            self._subscriber_id = None

    def _run_loop(self):
        """Consume events and refresh UI."""
        if not self._queue:
            return
        with Live(
            self._render_layout(),
            refresh_per_second=max(1, int(1 / self.refresh_rate)),
            console=self.console,
            transient=False,
        ) as live:
            while not self._stop_event.is_set():
                self._drain_queue()
                live.update(self._render_layout())
                time.sleep(self.refresh_rate)

    def _drain_queue(self):
        if not self._queue:
            return
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            self._module_order.setdefault(event.module_name, event.timestamp)
            self._buffers[event.module_name].append(event)

    def _render_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["header"].update(self._render_header())
        layout["body"].update(self._render_body())
        layout["footer"].update(self._render_footer())
        return layout

    def _render_header(self) -> RenderableType:
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="right")
        module_count = len(self._buffers)
        table.add_row(
            f"[bold]Pipeline Logs[/bold] • {module_count} module{'s' if module_count != 1 else ''}",
            "[dim]Press Ctrl+C to exit[/dim]",
        )
        return Panel(table, style="cyan")

    def _render_body(self) -> RenderableType:
        if not self._buffers:
            return Panel("[dim]Waiting for module output...[/dim]", border_style="dim")
        panels = []
        for module_name in sorted(self._module_order, key=self._module_order.get):
            events = self._buffers.get(module_name, [])
            lines = Text()
            for event in events:
                lines.append(
                    f"{event.formatted_timestamp()} ",
                    style="bold green" if event.stream == "stdout" else "bold red"
                    if event.stream == "stderr"
                    else "cyan",
                )
                lines.append(f"[{event.stream}] ", style="dim")
                lines.append(f"{event.message}\n", style="white")
            panels.append(Panel(lines or Text(""), title=module_name, border_style="blue"))
        return Columns(panels, expand=True, equal=True) if panels else Panel("")

    def _render_footer(self) -> RenderableType:
        footer = Table.grid(expand=True)
        footer.add_column()
        footer.add_row("[dim]Logs are persisted per module under the run logs directory.[/dim]")
        return Panel(footer, style="dim")

