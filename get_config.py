import yaml

DEFAULT_CONFIG_PATH = 'config.yaml'


class Config:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        with open(config_path, 'r') as stream:
            self.content = yaml.safe_load(stream)


config = Config()
