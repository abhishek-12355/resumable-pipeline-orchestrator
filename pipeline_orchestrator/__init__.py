"""Pipeline Orchestrator - YAML-driven pipeline orchestration with resource management."""

from pipeline_orchestrator.orchestrator import PipelineOrchestrator
from pipeline_orchestrator.module import BaseModule, FunctionModuleAdapter, ModuleLoader
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.config import PipelineConfig
from pipeline_orchestrator.exceptions import (
    PipelineOrchestratorError,
    ConfigurationError,
    DependencyError,
    ResourceError,
    ModuleExecutionError,
    CheckpointError,
    NestedExecutionError
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "PipelineOrchestrator",
    "BaseModule",
    "FunctionModuleAdapter",
    "ModuleLoader",
    "ModuleContext",
    "PipelineConfig",
    # Exceptions
    "PipelineOrchestratorError",
    "ConfigurationError",
    "DependencyError",
    "ResourceError",
    "ModuleExecutionError",
    "CheckpointError",
    "NestedExecutionError",
]

