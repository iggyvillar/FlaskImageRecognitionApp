"""Integration tests for model loading and prediction for Assignment 3"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import UnidentifiedImageError

import model
from model import predict_result, preprocess_img


# Happy Path Test
@patch("model.load_model")  # replaces the actual load_model function
def test_model_predict_success(mock_load_model):
    """
    Test that predict_result works when the model is loaded successfully.
    """
    dummy_model = MagicMock()  # fake model object

    # Pretend it predicts '4' because of highest probability
    dummy_model.predict.return_value = [
        [0.1, 0.05, 0.05, 0.05, 0.5, 0.05, 0.06, 0.09, 0.15, 0.2]
    ]
    mock_load_model.return_value = dummy_model

    # Patch the global model in model.py
    with patch("model.model", dummy_model):
        # Create a dummy image input
        import numpy as np

        dummy_img = np.zeros((1, 224, 224, 3))

        result = predict_result(dummy_img)
        assert result == 4  # Index of max probability

        # Ensure predict was called
        dummy_model.predict.assert_called_once_with(dummy_img)


# Sad Path Test
def test_preprocess_invalid_image_graceful(monkeypatch):
    """
    Test preprocess_img with a non-image input
    Make sure predict_result is never called
    and an exception is raised.
    """

    # Create a fake non-image file
    fake_file = io.BytesIO(b"not an image")

    # Spy setup
    called = {"flag": False}

    def fake_predict(img):
        called["flag"] = True

    # Use monkeypatch to replace predict_result with fake_predict
    monkeypatch.setattr(model, "predict_result", fake_predict)

    # Attempt to preprocess raises an error
    with pytest.raises(UnidentifiedImageError) as excinfo:
        preprocess_img(fake_file)

    # Ensure predict_result was never called
    assert called["flag"] is False
