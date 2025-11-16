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
                # Create checkpoint manager configuration
                config = CheckpointManagerConfig(
                    checkpoint_dir=str(self.checkpoint_directory)
                )
                
                # Initialize checkpoint manager
                self.manager = CheckpointManager(config=config)
            except Exception as e:
                raise CheckpointError(f"Failed to initialize checkpoint manager: {e}")
        else:
            self.manager = None
    
    def save_result(self, module_name: str, result: Any, is_error: bool = False) -> str:
        """
        Save module result as checkpoint.
        
        Args:
            module_name: Name of the module
            result: Result object or error to save
            is_error: Whether result is an error/exception
            
        Returns:
            Path to saved checkpoint file
            
        Raises:
            CheckpointError: If saving fails
        """
        if not self.enabled:
            raise CheckpointError("Checkpointing is disabled")
        
        checkpoint_name = f"{module_name}_result"
        
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
            status = "failed"
            description = f"Error from module {module_name}"
        else:
            data_to_save = result
            status = "completed" if not is_error else "failed"
            description = f"Result from module {module_name}"
        
        try:
            file_path = self.manager.save(
                name=checkpoint_name,
                data=data_to_save,
                description=description,
                custom_metadata={
                    "module_name": module_name,
                    "status": status,
                    "is_error": is_error
                }
            )
            return file_path
        except Exception as e:
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
            result, metadata = self.manager.load(checkpoint_name)
            return result, metadata
        except Exception as e:
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

