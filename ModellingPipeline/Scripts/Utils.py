import json

class ConfigHandler:
    def __init__(self, config_path):
        """Load configuration from a file (JSON or YAML)."""
        if config_path.endswith('.json'):
            with open(config_path, 'r') as file:
                self.config = json.load(file)
        else:
            raise ValueError("Unsupported configuration file format. Use JSON")

    def get(self, key, default=None):
        """Retrieve a configuration value."""
        return self.config.get(key, default)
    
    def fetch_config(self):
        if(self.config):
            return self.config
        else:
            raise ValueError("Could not read config file ,provide correct path.")
