"""
ACCEPTANCE TEST: AC1 - Upload and Predict
Feature: Verify that a user can upload a valid image and get a class index.

Given that the Flask app is running, verify that app exposes the /prediction route
When the user submits a image file to the prediction route
Then the app should return HTTP 200 and the predicted class index in the response

Test Steps:
  1. Create a small in-memory PNG file.
  2. Pretend the model always says "4" by stubbing the predict function
  3. Send the image.
  4. Assert HTTP 200 and that response should contain the predicted value.

Expected output:
  - HTTP 200 and response contains the predicted index "4".

Cleanup:
  - None.
"""

"""
ACCEPTANCE TEST: AC2 Invalid File Type uploaded
Feature: Verify the app rejects non image uploads.

GIVEN the Flask app is running,
WHEN the user uploads a non image file (e.g., .txt),
THEN the app should return HTTP 200 (or 400) and show an error message ("No file uploaded").

Test Steps:
  1. Create a small in memory text file.
  2. POST it as 'file'.
  3. Assert response status is 200.
  4. Assert response contains error message ("No file uploaded").

Expected Output:
  - Response contains error message, no prediction attempted.

Cleanup:
- None.
"""

from io import BytesIO
from unittest.mock import patch, MagicMock
import pytest

from app import app
import model


@pytest.fixture

def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def make_dummy_image_bytes():
    """
    Return bytes for a simulated image file.
    """
    img_data = BytesIO(b"fake_image_data")
    img_data.name = "test_image.jpeg"
    return img_data


def test_ac1_upload_and_predict(client):
    """
    AC1: Acceptance test.
    Upload a valid image and verify we get a prediction.
    """
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [
        [0.1, 0.05, 0.05, 0.05, 0.8, 0.05, 0.06, 0.09, 0.15, 0.2]
    ]

    with patch("model.load_model", return_value=dummy_model):
        with patch("model.model", dummy_model):
            img_file = make_dummy_image_bytes()
            img_file.name = "test_image.jpeg"
            response = client.post(
                "/prediction",
                data={"file": (img_file, img_file.name)},
                content_type="multipart/form-data",
            )

            assert response.status_code == 200
            assert b"4" in response.data


def test_ac2_invalid_file_type(client):
    """
    AC2: Acceptance test 2.
    Upload a non-image file (.txt) and verify the app rejects it.
    """
    txt_file = BytesIO(b"CSCN73010 Assignment 4.")
    txt_file.name = "test.txt"

    response = client.post(
        "/prediction",
        data={"file": (txt_file, txt_file.name)},
        content_type="multipart/form-data",
    )

    assert response.status_code in (200, 400)
    assert b"No file uploaded" in response.data or b"error" in response.data.lower()
