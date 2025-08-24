"""
Prediction pipeline orchestration.

This module instantiates the display and reading ROI model wrappers and exposes a
single `PredictionPipeline` that coordinates ROI detection flows.
"""

from typing import Tuple

import numpy as np

from metervision.constants import DIR_CONFIG_FILE, PARAMS_CONFIG_FILE
from metervision.models.ocr_model import TrOCRRecognizer
from metervision.models.roi_display import DisplayDetector
from metervision.models.roi_reading import ReadingDetector


class MeterVisionPipeline:
    """
    Orchestrates display and reading detectors to produce ROIs.

    Attributes
    ----------
    dir_config : object
        Path/config object loaded from your directory config (DIR_CONFIG_FILE).
    params_config : object
        Parameter object (PARAMS_CONFIG_FILE) with resizing and other params.
    display_detector : DisplayDetector
        Detector that finds the display ROI.
    reading_detector : ReadingDetector
        Detector that finds the reading ROI within the display ROI.
    """

    def __init__(self):
        self.dir_config = DIR_CONFIG_FILE
        self.params_config = PARAMS_CONFIG_FILE

        # instantiate the display-detector, reading-detector and TrOCR Recognizer with configured weights/params
        self.display_detector = DisplayDetector(
            model_path=self.dir_config.display_roi_model,
            params=self.params_config.display_roi_resized,
        )

        self.reading_detector = ReadingDetector(
            model_path=self.dir_config.reading_roi_model,
            params=self.params_config.reading_roi_resized,
        )

        self.trocr_recognizer = TrOCRRecognizer(
            model_source=self.dir_config.trocr_model
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Run the full prediction flow.

        Parameters
        ----------
        image  : ndarray
            Original image array.

        Returns
        -------
        Tuple[ndarray, ndarray]
            ((display_image, reading_image, recognize_reading) — extracted/resized images for display ROI and reading ROI
                                                                 and recognized text.
        """

        # Detect and extract the display ROI
        display_image = self.display_detector.extract_display_roi(img=image)

        # Detect and extract the reading ROI inside the display ROI
        reading_image = self.reading_detector.extract_reading_roi(img=display_image)

        # Recognize the Readings from the Reading Image
        recognize_reading = self.trocr_recognizer.recognize_reading(image=reading_image)
        if len(recognize_reading) == 0:
            recognize_reading = "No Reading Found"

        return display_image, reading_image, recognize_reading
