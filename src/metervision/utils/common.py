import yaml
from box import ConfigBox
from PIL import Image
import numpy as np
import sys

from metervision.logger.logs import logging
from metervision.exception.custom_exception import CustomException

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


# Reading the Images and returned as np.array
def read_img(img_path):
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        logging.info("Image Converted into Numpy Array")
        return img_array
    except Exception as e:
        CustomException(str(e), sys)
