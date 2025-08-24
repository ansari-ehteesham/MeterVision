import sys

import cv2

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging


def extract_roi(img_arr, coords, resized_shape, task):
    try:
        logging.info(f"Trying to Extract {task} ROI")
        if not coords.any():
            logging.warning("No Bounding Box Founded")
            return None
        x_coords = coords[:, 0, 0]
        y_coords = coords[:, 0, 1]
        x1, x2 = x_coords.min(), x_coords.max()
        y1, y2 = y_coords.min(), y_coords.max()

        display_only_img = img_arr[y1:y2, x1:x2]
        logging.info(f"{task} ROI Extraction Completed Successfully")

        resized_display_img = cv2.resize(
            display_only_img, resized_shape, interpolation=cv2.INTER_CUBIC
        )
        logging.info(f"{task} Extracted ROI Resized Completed")

        return resized_display_img
    except Exception as e:
        raise CustomException(str(e), sys)
