import yaml
from box import ConfigBox

from metervision.logger.logs import logging


def read_yaml(file_dir):
    with open(file_dir, "r") as f:
        result = yaml.safe_load(f)
        f.close()
    logging.info(f"YAML file Loaded Successfully")
    return ConfigBox(result)

