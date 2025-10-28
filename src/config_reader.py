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
    def _to_parts(self, v) -> list[str]:
        """Return a list of string parts for comma separated values."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        s = str(v).strip()
        # handle empty
        if s == '':
            return []
        parts = [p.strip() for p in s.split(',') if p.strip() != '']
        return parts
		
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

    def get_int_list(self, key: str, default = None) -> list[int]:
        v = self.get(key, default)
      #  print(f"get_int_list for key={key} raw value: {v}")
        parts = self._to_parts(v)
     #   print(f"get_int_list for key={key} parts: {parts}")
        result = []
        for p in parts:
     #       print(f"Parsing part '{p}' for key '{key}' to int")
            # allow floats that represent integers (e.g., "50.0")
            try:
                if '.' in p:
                    result.append(int(float(p)))
                else:
                    result.append(int(p))
            except ValueError:
                raise ValueError(f"Cannot convert config value '{p}' for key '{key}' to int")
        return result

    def get_str_list(self, key: str, default = None) -> list:
        """
        Return the config value for `key` as a list of strings.
        Handles:
          - already-parsed lists
          - single scalar
          - comma-separated string like 'adam, sgd'
        """
        v = self.get(key, default)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(',') if p.strip() != '']
            l = [str(p) for p in parts]
     #       print(f"get_int_list for key={key} returning list: {l}")
            return l
        return [str(v)]
# ...existing code...

    def get_float_list(self, key, default=None):
        v = self.get(key, default)
        if v is None:
            return []
        # Already a list (numbers or strings)
        if isinstance(v, list):
            return [float(x) for x in v]
        # If a numeric scalar (int/float)
        if isinstance(v, (int, float)):
            return [float(v)]
        # If a string that may be comma-separated
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(',') if p.strip() != '']
            l = [float(p) for p in parts]
     #       print(f"get_int_list for key={key} returning list: {l}")
            return l
            
        # Fallback
        return [float(v)]