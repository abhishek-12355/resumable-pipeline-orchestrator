"""Module interface, adapters, and loader for pipeline orchestrator."""

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from pipeline_orchestrator.exceptions import ConfigurationError
from pipeline_orchestrator.context import ModuleContext
from pipeline_orchestrator.logging_config import get_logger

logger = get_logger(__name__)


class BaseModule(ABC):
    """Base class for pipeline modules."""

    from enum import Enum

    class ModuleStatus(Enum):
        NOT_STARTED = "not_started"
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        FAILED = "failed"
        SUCCESS = "success"

    def __init__(self):
        self.status = BaseModule.ModuleStatus.NOT_STARTED

    @abstractmethod
    def run(self, context: ModuleContext) -> Any:
        """
        Execute the module.
        
        Args:
            context: Module execution context providing:
                - Dependency results
                - Nested execution API
                - Module metadata
            
        Returns:
            Result object (will be checkpointed automatically)
        """
        pass

    def set_status(self, status: "BaseModule.ModuleStatus"):
        self.status = status

    def get_status(self) -> "BaseModule.ModuleStatus":
        return self.status


class FunctionModuleAdapter(BaseModule):
    """Adapter for using standalone functions as modules."""
    
    def __init__(self, func: Callable):
        """
        Initialize function module adapter.
        
        Args:
            func: Function to execute as module
        """
        if not callable(func):
            raise ConfigurationError("Function must be callable")
        self.func = func
    
    def run(self, context: ModuleContext) -> Any:
        """
        Execute the function with context.
        
        Args:
            context: Module execution context
            
        Returns:
            Function result
        """
        return self.func(context)


class ModuleLoader:
    """Loader for importing and instantiating modules."""
    
    @staticmethod
    def load_module(module_config: Dict[str, Any]) -> BaseModule:
        """
        Load module from configuration.
        
        Supports two modes:
        1. Class-based: "path:ClassName" (e.g., "mymodules.module1:MyModule")
        2. Function-based: "script:function_name" (e.g., "path/to/script.py:function_name")
        
        Args:
            module_config: Module configuration dictionary
            
        Returns:
            BaseModule instance
            
        Raises:
            ConfigurationError: If module cannot be loaded
        """
        # Check for class-based module (path)
        if "path" in module_config:
            path = module_config["path"]
            return ModuleLoader._load_class_module(path)
        
        # Check for function-based module (script)
        if "script" in module_config:
            script_path = module_config["script"]
            return ModuleLoader._load_function_module(script_path)
        
        raise ConfigurationError(
            f"Module config must have either 'path' or 'script': {module_config}"
        )
    
    @staticmethod
    def _load_class_module(path: str) -> BaseModule:
        """
        Load class-based module from import path.
        
        Args:
            path: Import path in format "module.path:ClassName"
            
        Returns:
            BaseModule instance
        """
        if ":" not in path:
            raise ConfigurationError(
                f"Invalid module path format: '{path}'. "
                f"Expected format: 'module.path:ClassName'"
            )
        
        module_path, class_name = path.rsplit(":", 1)
        
        logger.debug(f"Loading class-based module: {module_path}.{class_name}")
        try:
            # Import module
            module = importlib.import_module(module_path)
            logger.debug(f"Imported module: {module_path}")
            
            # Get class
            if not hasattr(module, class_name):
                logger.error(f"Class '{class_name}' not found in module '{module_path}'")
                raise ConfigurationError(
                    f"Class '{class_name}' not found in module '{module_path}'"
                )
            
            cls = getattr(module, class_name)
            
            # Instantiate (assuming no required arguments)
            try:
                instance = cls()
                logger.debug(f"Instantiated class: {class_name}")
            except TypeError as e:
                logger.error(f"Failed to instantiate class '{class_name}': {e}")
                raise ConfigurationError(
                    f"Failed to instantiate class '{class_name}': {e}. "
                    f"Class should have no required arguments or accept context parameter."
                )
            
            # Check if it's a BaseModule or has run method
            if isinstance(instance, BaseModule):
                logger.debug(f"Module {class_name} is a BaseModule")
                return instance
            elif hasattr(instance, "run") and callable(getattr(instance, "run")):
                # Wrap in adapter if it's not BaseModule but has run method
                logger.debug(f"Wrapping module {class_name} with adapter")
                return _ModuleWrapper(instance)
            else:
                logger.error(
                    f"Class '{class_name}' must be a subclass of BaseModule or have a 'run' method"
                )
                raise ConfigurationError(
                    f"Class '{class_name}' must be a subclass of BaseModule or have a 'run' method"
                )
        
        except ImportError as e:
            logger.error(f"Failed to import module '{module_path}': {e}")
            raise ConfigurationError(f"Failed to import module '{module_path}': {e}")
        except Exception as e:
            logger.error(f"Failed to load module from path '{path}': {e}")
            raise ConfigurationError(f"Failed to load module from path '{path}': {e}")
    
    @staticmethod
    def _load_function_module(script_path: str) -> BaseModule:
        """
        Load function-based module from script file.
        
        Args:
            script_path: Script path in format "path/to/script.py:function_name"
            
        Returns:
            FunctionModuleAdapter instance
        """
        if ":" not in script_path:
            raise ConfigurationError(
                f"Invalid script path format: '{script_path}'. "
                f"Expected format: 'path/to/script.py:function_name'"
            )
        
        file_path, function_name = script_path.rsplit(":", 1)
        
        script_path_obj = Path(file_path)
        if not script_path_obj.exists():
            logger.error(f"Script file not found: {file_path}")
            raise ConfigurationError(f"Script file not found: {file_path}")
        
        logger.debug(f"Loading function-based module: {file_path}:{function_name}")
        try:
            # Add script directory to path
            script_dir = str(script_path_obj.parent)
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
                logger.debug(f"Added {script_dir} to sys.path")
            
            # Load module from file
            module_name = script_path_obj.stem
            spec = importlib.util.spec_from_file_location(module_name, str(script_path_obj))
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load spec for script: {file_path}")
                raise ConfigurationError(f"Failed to load spec for script: {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.debug(f"Loaded script module: {module_name}")
            
            # Get function
            if not hasattr(module, function_name):
                logger.error(f"Function '{function_name}' not found in script '{file_path}'")
                raise ConfigurationError(
                    f"Function '{function_name}' not found in script '{file_path}'"
                )
            
            func = getattr(module, function_name)
            
            if not callable(func):
                logger.error(f"'{function_name}' in script '{file_path}' is not callable")
                raise ConfigurationError(
                    f"'{function_name}' in script '{file_path}' is not callable"
                )
            
            logger.debug(f"Loaded function: {function_name}")
            return FunctionModuleAdapter(func)
        
        except Exception as e:
            logger.error(f"Failed to load function from script '{script_path}': {e}")
            raise ConfigurationError(f"Failed to load function from script '{script_path}': {e}")


class _ModuleWrapper(BaseModule):
    """Wrapper for modules that have run method but don't inherit BaseModule."""
    
    def __init__(self, instance: Any):
        """
        Initialize module wrapper.
        
        Args:
            instance: Module instance with run method
        """
        self.instance = instance
    
    def run(self, context: ModuleContext) -> Any:
        """
        Execute wrapped module.
        
        Args:
            context: Module execution context
            
        Returns:
            Module result
        """
        # Try calling run with context
        try:
            return self.instance.run(context)
        except TypeError:
            # Try calling without context if context not accepted
            return self.instance.run()

