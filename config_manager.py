"""
Configuration management module for the recommendation system.
Handles loading and managing configuration settings from YAML files.
"""

import yaml
from typing import Any, Dict
import os

class ConfigManager:
    """
    Configuration manager for the recommendation engine.
    Provides structured access to configuration settings stored in YAML format.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file is not found
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """
        Get model architecture configuration.
        
        Returns:
            Dict containing model architecture parameters (layers, dimensions, etc.)
        """
        return self._config['model']
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """
        Get training parameters configuration.
        
        Returns:
            Dict containing training parameters (batch size, epochs, etc.)
        """
        return self._config['training']
    
    @property
    def regularization_config(self) -> Dict[str, Any]:
        """
        Get regularization configuration.
        
        Returns:
            Dict containing regularization parameters (dropout rates, L1/L2, etc.)
        """
        return self._config['regularization']
    
    @property
    def evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration.
        
        Returns:
            Dict containing evaluation parameters (metrics, test split, etc.)
        """
        return self._config['evaluation']
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.
        
        Returns:
            Dict containing preprocessing parameters (scaling, encoding, etc.)
        """
        return self._config['preprocessing']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key (str): Dot-separated configuration key (e.g., 'model.embedding_dim')
            default: Default value if key is not found
        
        Returns:
            Configuration value or default if not found
            
        Example:
            >>> config.get('model.embedding_dim', 64)
            128
        """
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            key (str): Dot-separated configuration key
            value: New value to set
            
        Example:
            >>> config.update('model.embedding_dim', 256)
        """
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value 