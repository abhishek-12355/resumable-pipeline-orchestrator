"""Custom exceptions for pipeline orchestrator."""


class PipelineOrchestratorError(Exception):
    """Base exception for pipeline orchestrator errors."""
    pass


class ConfigurationError(PipelineOrchestratorError):
    """Raised when pipeline configuration is invalid."""
    pass


class DependencyError(PipelineOrchestratorError):
    """Raised when module dependency is not found or not completed."""
    pass


class ResourceError(PipelineOrchestratorError):
    """Raised when insufficient resources are available for module/task execution."""
    pass


class ModuleExecutionError(PipelineOrchestratorError):
    """Raised when module execution fails."""
    pass


class CheckpointError(PipelineOrchestratorError):
    """Raised when checkpoint save/load operations fail."""
    pass


class NestedExecutionError(PipelineOrchestratorError):
    """Raised when nested task execution fails."""
    pass

