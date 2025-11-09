"""
Acceptance test cases for negative scenarios.
Tests application behavior when invalid or missing
"""
# test_acceptance_sad.py

import pytest
from app import app


@pytest.fixture
def client():
    """
    Fixture for the Flask test client.

    Purpose:
        Set up a test client for making requests to the Flask app during testing.

    Usage:
        Provides a `client` object to use for HTTP request simulations.
    """
    with app.test_client() as client_instance:
        yield client_instance


def test_acceptance_missing_file(client):
    """
    Test Case: No File Uploaded

    Purpose:
        Validate the application's behavior when no file is provided in the upload request.

    Scenario:
        - Simulate a POST request to the `/prediction` route with no file data.
        - Assert the response status code is 200 (to indicate a valid request was processed).
        - Verify that the response includes an appropriate error message.
    """
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data"
    )

    # Assertions
    assert response.status_code == 200
    assert b"File cannot be processed" in response.data
