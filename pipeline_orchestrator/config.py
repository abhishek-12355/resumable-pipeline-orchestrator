"""Configuration parsing and validation for pipeline orchestrator."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from pipeline_orchestrator.exceptions import ConfigurationError


class PipelineConfig:
    """Pipeline configuration parsed from YAML."""
    
    def __init__(
        self,
        name: str,
        execution: Dict[str, Any],
        resources: Dict[str, int],
        modules: List[Dict[str, Any]],
        checkpoint: Optional[Dict[str, Any]] = None,
        logging: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.execution = execution
        self.resources = resources
        self.modules = modules
        self.checkpoint = checkpoint or {}
        self.logging = logging or {}
        
        # Validate configuration
        self._validate()
    
    @classmethod
    def from_yaml_file(cls, config_path: str) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML file: {e}")
        except IOError as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "PipelineConfig":
        """
        Load configuration from dictionary.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            PipelineConfig instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config_data, dict) or "pipeline" not in config_data:
            raise ConfigurationError("Configuration must have 'pipeline' key")
        
        pipeline_config = config_data["pipeline"]
        
        # Extract name
        name = pipeline_config.get("name", "default_pipeline")
        
        # Extract execution config
        execution = pipeline_config.get("execution", {})
        execution_mode = execution.get("mode", "parallel")
        if execution_mode not in ("sequential", "parallel"):
            raise ConfigurationError(f"Invalid execution mode: {execution_mode}")
        
        worker_type = execution.get("worker_type", "process")
        if worker_type not in ("thread", "process"):
            raise ConfigurationError(f"Invalid worker type: {worker_type}")
        
        failure_policy = execution.get("failure_policy", "fail_fast")
        if failure_policy not in ("fail_fast", "collect_all"):
            raise ConfigurationError(f"Invalid failure policy: {failure_policy}")
        
        max_nested_depth = execution.get("max_nested_depth")
        
        execution_config = {
            "mode": execution_mode,
            "worker_type": worker_type,
            "failure_policy": failure_policy,
            "max_nested_depth": max_nested_depth
        }
        
        # Extract resources config
        resources_config = pipeline_config.get("resources", {})
        max_cpus = resources_config.get("max_cpus")
        max_gpus = resources_config.get("max_gpus")
        
        if max_cpus is not None and max_cpus <= 0:
            raise ConfigurationError("max_cpus must be positive")
        if max_gpus is not None and max_gpus < 0:
            raise ConfigurationError("max_gpus must be non-negative")
        
        resources = {
            "max_cpus": max_cpus,
            "max_gpus": max_gpus
        }
        
        # Extract checkpoint config
        checkpoint_config = pipeline_config.get("checkpoint", {})
        checkpoint_enabled = checkpoint_config.get("enabled", True)
        checkpoint_directory = checkpoint_config.get("directory", "./.checkpoints")
        
        checkpoint = {
            "enabled": checkpoint_enabled,
            "directory": checkpoint_directory
        }
        
        # Extract logging config
        logging_config = pipeline_config.get("logging", {})
        enable_live_logs = logging_config.get("enable_live_logs", True)
        logs_directory = logging_config.get("logs_directory", "./logs")
        max_log_file_bytes = logging_config.get("max_log_file_bytes", 10 * 1024 * 1024)
        log_backup_count = logging_config.get("log_backup_count", 5)
        
        if max_log_file_bytes is not None and max_log_file_bytes <= 0:
            raise ConfigurationError("max_log_file_bytes must be positive if provided")
        if log_backup_count is not None and log_backup_count < 0:
            raise ConfigurationError("log_backup_count must be non-negative")
        
        logging_settings = {
            "enable_live_logs": bool(enable_live_logs),
            "logs_directory": logs_directory,
            "max_log_file_bytes": max_log_file_bytes,
            "log_backup_count": log_backup_count,
        }
        
        # Extract modules
        modules = pipeline_config.get("modules", [])
        if not modules:
            raise ConfigurationError("Pipeline must have at least one module")
        
        # Validate modules
        for i, module in enumerate(modules):
            if not isinstance(module, dict):
                raise ConfigurationError(f"Module {i} must be a dictionary")
            
            if "name" not in module:
                raise ConfigurationError(f"Module {i} must have a 'name'")
            
            if "path" not in module and "script" not in module:
                raise ConfigurationError(f"Module {i} must have either 'path' or 'script'")
            
            if "path" in module and "script" in module:
                raise ConfigurationError(f"Module {i} cannot have both 'path' and 'script'")
            
            depends_on = module.get("depends_on", [])
            if not isinstance(depends_on, list):
                raise ConfigurationError(f"Module {i} 'depends_on' must be a list")
            
            module_resources = module.get("resources", {})
            module_cpus = module_resources.get("cpus", 1)
            module_gpus = module_resources.get("gpus", 0)
            
            if module_cpus <= 0:
                raise ConfigurationError(f"Module {i} CPUs must be positive")
            if module_gpus < 0:
                raise ConfigurationError(f"Module {i} GPUs must be non-negative")
        
        return cls(
            name=name,
            execution=execution_config,
            resources=resources,
            modules=modules,
            checkpoint=checkpoint,
            logging=logging_settings
        )
    
    def _validate(self):
        """Validate configuration."""
        # Validate module names are unique
        module_names = [m["name"] for m in self.modules]
        if len(module_names) != len(set(module_names)):
            raise ConfigurationError("Module names must be unique")
        
        # Validate dependencies reference existing modules
        for module in self.modules:
            for dep in module.get("depends_on", []):
                if dep not in module_names:
                    raise ConfigurationError(
                        f"Module '{module['name']}' depends on non-existent module '{dep}'"
                    )
        
        # Validate no circular dependencies (basic check)
        # More thorough check done in DependencyGraph
        visited = set()
        
        def has_cycle(module_name: str, path: set) -> bool:
            if module_name in path:
                return True
            if module_name in visited:
                return False
            
            visited.add(module_name)
            path.add(module_name)
            
            module = next(m for m in self.modules if m["name"] == module_name)
            for dep in module.get("depends_on", []):
                if has_cycle(dep, path.copy()):
                    return True
            
            return False
        
        for module in self.modules:
            if has_cycle(module["name"], set()):
                raise ConfigurationError(f"Circular dependency detected involving module '{module['name']}'")
    
    def get_module_config(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Module configuration dictionary or None if not found
        """
        for module in self.modules:
            if module["name"] == module_name:
                return module
        return None

