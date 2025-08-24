"""
Streamlit application entrypoint for MeterVision.

This module wires up the Streamlit UI (pages) and loads the prediction pipeline once
using Streamlit's resource caching. It also loads environment variables and a CSS
file for styling the UI.

Notes:
- Keeps behavior unchanged; only adds documentation and inline comments.
- Uses st.session_state to ensure .env is loaded once per Streamlit session.
"""

import sys

import cv2
import streamlit as st
from dotenv import load_dotenv

from metervision.constants import DIR_CONFIG_FILE
from metervision.exception.custom_exception import CustomException
# Own Module
from metervision.logger.logs import logging
from metervision.pipeline.predictor import MeterVisionPipeline
from metervision.utils.file_utils import read_img

# ------------------------------------------------------------------------------------------------------------------
# Loading Environment Variables
# ------------------------------------------------------------------------------------------------------------------


try:
    if "env_loaded" not in st.session_state:
        load_dotenv()
        st.session_state["env_loaded"] = True
        logging.info("Environment Variables are Loaded Successfully")
    else:
        pass
except Exception as e:
    raise CustomException(str(e), sys)


# ------------------------------------------------------------------------------------------------------------------
# Streamlit page configuration and CSS
# ------------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="MeterVision",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open(DIR_CONFIG_FILE.sidebar_css) as f:
        css = f.read()
    st.markdown(f"<style> {css} </style>", unsafe_allow_html=True)
except Exception as e:
    raise CustomException(str(e), sys)


# ------------------------------------------------------------------------------------------------------------------
# Model(s) loading (cached resource)
# ------------------------------------------------------------------------------------------------------------------


@st.cache_resource
def load_pipeline() -> MeterVisionPipeline:
    """
    Load models and return a ready-to-use PredictionPipeline instance.

    Decorating with `st.cache_resource` ensures heavy model loading happens only once
    per app session & worker, avoiding repeated expensive initializations.
    """

    pipeline = MeterVisionPipeline()
    return pipeline


pipeline = load_pipeline()


# ------------------------------------------------------------------------------------------------------------------
# Home Page
# ------------------------------------------------------------------------------------------------------------------


def home_page() -> None:
    """
    Home page layout.

    Renders the main header and a short sub-header.
    """

    with st.container():
        st.markdown(
            "<div class='main-header'>⚡ MeterVision Dashboard</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sub-header'>A Vision to Power the Electric Meter</div>",
            unsafe_allow_html=True,
        )
    return None


# ------------------------------------------------------------------------------------------------------------------
# Bulk Data Uplaoder Page
# ------------------------------------------------------------------------------------------------------------------


def bulk_data_page() -> None:

    st.title("Bulk Data")
    return None


# ------------------------------------------------------------------------------------------------------------------
# Model Prediction Page
# ------------------------------------------------------------------------------------------------------------------


def prediction_page() -> None:
    """
    Prediction page UI and flow.

    - Accepts an image upload (jpg/jpeg/png).
    - On 'Start Prediction' button press: reads the image, runs the prediction pipeline,
      and displays original + ROI images and a (disabled) number input for the reading.
    """

    st.title("Prediction Page")
    uploaded_file = st.file_uploader(
        label="Select the Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        logging.info(f"Meter Image has been Uploaded")

        # Start prediction only when user clicks the button
        if st.button("Start Prediction", type="primary"):
            logging.info("Prediction has been Started....")

            # Use spinner to inform the user the app is working
            with st.spinner("Wait For It...", show_time=True):

                # read_img should return an image array (np.ndarray)
                original_image = read_img(uploaded_file)

                # run the pipeline; returns display_arr and reading_arr (image arrays)
                display_image, reading_image, readings = pipeline.predict(
                    image=original_image
                )

            # Layout: show original image and ROI images in columns
            with st.container(height=400):
                img_col, roi_cols = st.columns(spec=2, gap="small", border=True)

                # Guard: check arrays contain data before attempting to display
                # .any() used to be consistent with your original check; keep as-is.
                if (
                    getattr(display_image, "any", lambda: False)()
                    and getattr(reading_image, "any", lambda: False)()
                ):
                    with img_col:
                        st.image(
                            image=cv2.resize(original_image, (300, 300)),
                            caption="Meter Image",
                        )
                    with roi_cols:
                        st.image(image=display_image, caption="ROI Image", width=300)
                        st.image(image=reading_image, caption="ROI Image", width=300)

            # Show reading as a disabled number input (placeholder — you can fill value)
            with st.container():
                st.text_input(label="Meter Reading", value=readings, disabled=True)

    return None


# ------------------------------------------------------------------------------------------------------------------
# Model Training Page
# ------------------------------------------------------------------------------------------------------------------


def model_training_page() -> None:
    st.title("Model Training")
    return None


# ------------------------------------------------------------------------------------------------------------------
# Navigation Page Setup
# ------------------------------------------------------------------------------------------------------------------
pg = st.navigation(
    [
        st.Page(home_page, title="🏠 Home Page", default=True),
        st.Page(bulk_data_page, title="📂 Bulk Data"),
        st.Page(prediction_page, title="🔍 Prediction"),
        st.Page(model_training_page, title="⚙️ Model Training"),
    ]
)
pg.run()
