"""
TrOCR recognizer wrapper for MeterVision.

This module loads a TrOCR processor + VisionEncoderDecoderModel from a directory
or model id and exposes a simple `TrOCRRecognizer` class for text recognition.

Features:
- Automatically uses GPU if available, otherwise CPU.
- Moves model and tensors to the selected device.
- Uses `torch.inference_mode()` to reduce overhead during generation.
- Exposes `generate_kwargs` to tune generation for speed/quality tradeoffs
  (e.g., num_beams=1 for greedy decoding which is faster).
- Optionally limits PyTorch thread usage on CPU to improve single-request latency.
"""

import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging


class TrOCRRecognizer:
    """
    High-level wrapper around TrOCR processor & model for recognizing text from images.

    Parameters
    ----------
    model_source : str
        Path to the pretrained model directory or model identifier (e.g. HuggingFace name).
    generate_kwargs : Optional[Dict[str, Any]]
        Keyword arguments passed to `model.generate(...)`. If None, defaults are used for
        faster greedy decoding.
    cpu_num_threads : Optional[int]
        If device is CPU, optionally set the number of PyTorch intra-op threads to
        limit parallelism (can improve latency for single-image inference).
    """

    def __init__(
        self, model_source: str, generate_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.model_source = model_source
        self.generate_kwargs = generate_kwargs or {
            # Greedy decoding (fast) — increase num_beams for quality at cost of speed
            "max_length": 128,
            "num_beams": 1,
            "do_sample": False,
            "early_stopping": True,
        }

        try:
            # Determine device (GPU if available)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("CUDA available — using GPU for OCR inference.")
            else:
                self.device = torch.device("cpu")
                logging.info(
                    "GPU not available — falling back to CPU for OCR inference."
                )

            # Use from_pretrained for compatibility with both local folders and HF hub ids
            self.processor = TrOCRProcessor.from_pretrained(model_source)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_source)

            # Move model to device (if torch available)
            if self.device is not None:
                self.model.to(self.device)

            logging.info("TrOCR processor and model loaded successfully.")
        except Exception as exc:
            raise CustomException(str(exc), sys)

    def recognize_reading(self, image: np.ndarray) -> str:
        """
        Recognize Reading from a single image.

        Parameters
        ----------
        image : np.ndarray
            Input Image

        Returns
        -------
        str
            Recognized text (first result from batch_decode).
        """
        try:
            # Preprocess images -> pixel_values (tensor)
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            with torch.inference_mode():

                if self.device.type == "cuda":
                    try:
                        from torch.amp import autocast

                        with autocast():
                            generated_ids = self.model.generate(
                                pixel_values, **self.generate_kwargs
                            )
                    except Exception:

                        generated_ids = self.model.generate(
                            pixel_values, **self.generate_kwargs
                        )
                else:
                    generated_ids = self.model.generate(
                        pixel_values, **self.generate_kwargs
                    )

            # Decode predicted ids to text
            generated_reading = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return generated_reading

        except Exception as exc:
            raise CustomException(f"OCR recognition failed: {exc}", sys)
