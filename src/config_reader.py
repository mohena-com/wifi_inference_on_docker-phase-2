import os
import ast
from pathlib import Path

class ConfigReader:
    def __init__(self, config_path):
        self.config = {}
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from properties file."""
        with open(self.config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    self.config[key.strip()] = value.strip()
    
    def get(self, key, default=None):
        """Get configuration value with optional default."""
        return self.config.get(key, default)
    
    def get_int(self, key, default=None):
        """Get integer configuration value."""
        value = self.get(key, default)
        return int(value) if value is not None else default
    
    def get_float(self, key, default=None):
        """Get float configuration value."""
        value = self.get(key, default)
        return float(value) if value is not None else default
    
    def get_bool(self, key, default=None):
        """Get boolean configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return value.lower() == 'true'
    
    def get_list(self, key, default=None, type_func=str):
        """Get list configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return [type_func(x.strip()) for x in value.split(',')]
    
    def get_tuple(self, key, default=None, type_func=int):
        """Get tuple configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return tuple(type_func(x.strip()) for x in value.split(','))
    
    def get_path(self, key, default=None):
        """Get path configuration value."""
        value = self.get(key, default)
        if value is None:
            return default
        return Path(value) 