import yaml


def read_config(config_file: str):
    with open(config_file, "r") as file:
        configs = yaml.safe_load(file)
        return configs
