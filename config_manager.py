import yaml
from typing import Any, Dict
import os

class ConfigManager:
    """Manages configuration settings for the recommendation engine."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model architecture configuration."""
        return self._config['model']
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training parameters configuration."""
        return self._config['training']
    
    @property
    def regularization_config(self) -> Dict[str, Any]:
        """Get regularization configuration."""
        return self._config['regularization']
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config['evaluation']
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config['preprocessing']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Dot-separated configuration key (e.g., 'model.embedding_dim')
            default: Default value if key is not found
        
        Returns:
            Configuration value
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            key: Dot-separated configuration key
            value: New value
        """
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value 