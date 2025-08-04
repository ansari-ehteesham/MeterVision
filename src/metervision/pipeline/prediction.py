from metervision.constant import (DIR_CONFIG_FILE, PARAMS_CONFIG_FILE)

from metervision.models.roi_model import ROIModel

class PredictionPipeline:
    def __init__(self):
        self.dir_config = DIR_CONFIG_FILE
        self.params_config = PARAMS_CONFIG_FILE

        self.roi = ROIModel(
            model_path=self.dir_config.roi_model,
            params = self.params_config.roi_resized
        )

    def prediction(self, img_array):
        # ROI Prediction
        bb_img_arr, display_arr = self.roi.roi_prediction(img_arr=img_array)

        return bb_img_arr, display_arr