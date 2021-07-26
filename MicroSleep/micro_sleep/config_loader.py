import configparser


def load_config(config_file):
    return Config(config_file).app_config


class Config:
    class __ConfigInit:
        def __init__(self, file_name):
            self.config = configparser.ConfigParser()
            self.config.read(file_name)
            self.app_config = LoadConfig(self.config)

    instance = None

    def __init__(self, file_name):
        if not Config.instance:
            Config.instance = Config.__ConfigInit(file_name)

    def __getattr__(self, name):
        return getattr(self.instance, name)


class LoadConfig:
    def __init__(self, config):

        try:
            self.video_path = config['DATASOURCE']['video_path']
            self.model_path = config['DATASOURCE']['model_path']

        except Exception as e:
            raise Exception("Error loading config. Error: {}".format(e))