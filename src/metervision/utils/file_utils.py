import json
import sys

import numpy as np
import yaml
from box import ConfigBox
from PIL import Image

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging


# Reading the YAML Files
def read_yaml(file_dir):
    try:
        with open(file_dir, "r") as f:
            result = yaml.safe_load(f)
            f.close()
        logging.info(f"YAML file Loaded Successfully")
        return ConfigBox(result)
    except Exception as e:
        raise CustomException(str(e), sys)


# Reading the JSON File
def read_json(file_dir):
    try:
        with open(file_dir, "r") as f:
            result = json.load()
            f.close()
        logging.info(f"JSON file Loaded Successfully")
        return result
    except Exception as e:
        raise CustomException(str(e), sys)


# Reading the Images and returned as np.array
def read_img(img_path):
    try:
        img = Image.open(img_path)
        img = img.convert("RGB")
        img_array = np.array(img, dtype=np.uint8)
        logging.info("Image Converted into Numpy Array")
        return img_array
    except Exception as e:
        CustomException(str(e), sys)
