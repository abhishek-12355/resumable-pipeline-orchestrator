"""Checkpoint integration using checkpoint-manager (internal to orchestrator)."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from checkpoint_manager import CheckpointManager, CheckpointManagerConfig
except ImportError:
    raise ImportError(
        "checkpoint-manager is required. Install with: pip install checkpoint-manager"
    )

from pipeline_orchestrator.exceptions import CheckpointError
from pipeline_orchestrator.logging_config import get_logger
from pipeline_orchestrator.module import BaseModule

logger = get_logger("orchestrator")


class PipelineCheckpointManager:
    """Wrapper around checkpoint-manager for pipeline checkpointing."""
    
    def __init__(self, checkpoint_directory: str, enabled: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_directory: Directory for storing checkpoints
            enabled: Whether checkpointing is enabled
        """
        self.enabled = enabled
        self.checkpoint_directory = Path(checkpoint_directory).resolve()
        
        if self.enabled:
            try:
                logger.info(f"Initializing checkpoint manager: {self.checkpoint_directory}")
                # Create checkpoint manager configuration
                config = CheckpointManagerConfig(
                    checkpoint_dir=str(self.checkpoint_directory)
                )
                
                # Initialize checkpoint manager
                self.manager = CheckpointManager(config=config)
                
                logger.debug("Checkpoint manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize checkpoint manager: {e}")
                raise CheckpointError(f"Failed to initialize checkpoint manager: {e}")
        else:
            self.manager = None
            logger.debug("Checkpoint manager disabled")
    
    def save_result(
        self, 
        module_name: str, 
        result: Any, 
        is_error: bool = False,
        status: Optional[BaseModule.ModuleStatus] = None
    ) -> str:
        """
        Save module result as checkpoint.
        
        Args:
            module_name: Name of the module
            result: Result object or error to save
            is_error: Whether result is an error/exception
            status: ModuleStatus enum value (if None, derived from is_error)
            
        Returns:
            Path to saved checkpoint file
            
        Raises:
            CheckpointError: If saving fails
        """
        if not self.enabled:
            raise CheckpointError("Checkpointing is disabled")
        
        checkpoint_name = f"{module_name}_result"
        
        # Determine status
        if status is not None:
            # Use provided status enum value (convert to lowercase string)
            status_str = status.value
        elif is_error:
            status_str = "failed"
        else:
            status_str = "success"
        
        # Prepare error data if it's an exception
        if is_error and isinstance(result, Exception):
            error_data = {
                "error_type": type(result).__name__,
                "error_message": str(result),
                "error_repr": repr(result)
            }
            # Try to serialize exception if possible
            try:
                import pickle
                error_data["error_pickled"] = pickle.dumps(result)
            except Exception:
                pass
            data_to_save = error_data
            description = f"Error from module {module_name}"
        else:
            data_to_save = result
            description = f"Result from module {module_name}"
        
        try:
            logger.debug(f"Saving checkpoint for {module_name} (status={status_str}, is_error={is_error})")
            file_path = self.manager.save(
                name=checkpoint_name,
                data=data_to_save,
                description=description,
                custom_metadata={
                    "module_name": module_name,
                    "status": status_str,
                    "is_error": is_error
                }
            )
            logger.debug(f"Checkpoint saved for {module_name}: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {module_name}: {e}")
            raise CheckpointError(f"Failed to save checkpoint for module '{module_name}': {e}")
    
    def load_result(self, module_name: str) -> Tuple[Any, Dict]:
        """
        Load module result from checkpoint.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Tuple of (result, metadata)
            
        Raises:
            CheckpointError: If checkpoint not found or loading fails
        """
        if not self.enabled:
            raise CheckpointError("Checkpointing is disabled")
        
        checkpoint_name = f"{module_name}_result"
        
        try:
            logger.debug(f"Loading checkpoint for {module_name}")
            result, metadata = self.manager.load(checkpoint_name)
            logger.debug(f"Checkpoint loaded for {module_name}")
            return result, metadata
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {module_name}: {e}")
            raise CheckpointError(
                f"Failed to load checkpoint for module '{module_name}': {e}"
            )
    
    def has_checkpoint(self, module_name: str) -> bool:
        """
        Check if checkpoint exists for a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        if not self.enabled:
            return False
        
        checkpoint_name = f"{module_name}_result"
        
        try:
            checkpoints = self.manager.list_checkpoints()
            return checkpoint_name in checkpoints
        except Exception:
            return False
    
    def get_checkpoint_metadata(self, module_name: str) -> Optional[Dict]:
        """
        Get checkpoint metadata without loading data.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Metadata dictionary or None if checkpoint doesn't exist
        """
        if not self.enabled:
            return None
        
        checkpoint_name = f"{module_name}_result"
        
        try:
            metadata = self.manager.get_metadata(checkpoint_name)
            return metadata
        except Exception:
            return None
    
    def list_completed_modules(self) -> list:
        """
        List all modules that have checkpoints.
        
        Returns:
            List of module names with checkpoints
        """
        if not self.enabled:
            return []
        
        try:
            checkpoints = self.manager.list_checkpoints()
            # Extract module names from checkpoint names (remove "_result" suffix)
            completed = []
            for checkpoint_name in checkpoints:
                if checkpoint_name.endswith("_result"):
                    module_name = checkpoint_name[:-7]  # Remove "_result"
                    completed.append(module_name)
            return completed
        except Exception:
            return []
    
    def cleanup_module_checkpoint(self, module_name: str):
        """
        Delete checkpoint for a module.
        
        Args:
            module_name: Name of the module
        """
        if not self.enabled:
            return
        
        checkpoint_name = f"{module_name}_result"
        
        try:
            self.manager.cleanup(checkpoint_name)
        except Exception:
            # Ignore errors during cleanup
            pass
    
    def cleanup_all_checkpoints(self):
        """Delete all checkpoints."""
        if not self.enabled:
            return
        
        try:
            self.manager.cleanup("all")
        except Exception:
            # Ignore errors during cleanup
            pass
    
    def get_module_status(self, module_name: str) -> Optional[BaseModule.ModuleStatus]:
        """
        Get module status from checkpoint metadata.
        
        Args:
            module_name: Name of the module
            
        Returns:
            ModuleStatus enum value or None if checkpoint doesn't exist
        """
        if not self.enabled:
            return None
        
        metadata = self.get_checkpoint_metadata(module_name)
        if metadata is None:
            return None
        
        # Extract status from metadata
        custom_metadata = metadata.get("custom_metadata", {})
        status_str = custom_metadata.get("status")
        
        if status_str is None:
            # Fallback: derive from is_error flag
            is_error = custom_metadata.get("is_error", False)
            status_str = "failed" if is_error else "success"
        
        # Map status string to enum
        # Handle backward compatibility: "completed" -> "success"
        if status_str == "completed":
            status_str = "success"
        
        # Map to enum value
        status_map = {
            "not_started": BaseModule.ModuleStatus.NOT_STARTED,
            "pending": BaseModule.ModuleStatus.PENDING,
            "in_progress": BaseModule.ModuleStatus.IN_PROGRESS,
            "failed": BaseModule.ModuleStatus.FAILED,
            "success": BaseModule.ModuleStatus.SUCCESS,
        }
        
        return status_map.get(status_str)

