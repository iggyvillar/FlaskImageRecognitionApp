"""
Assignment 4 - Demonstration Test File
This test intentionally fails to show pipeline behavior
"""

import pytest


def test_pipeline_stops_on_failure():
    """
    AC: Demonstrate that failing unit tests prevent acceptance tests from running
    
    Given: The CI/CD pipeline has 4 sequential jobs
    When: A unit test fails
    Then: Integration and acceptance tests should be skipped
    """
    # INTENTIONAL FAILURE FOR DEMO
    assert False, "This failure should stop the pipeline"


def test_this_one_passes():
    """
    AC: This test demonstrates partial success
    
    Some tests can pass while others fail
    """
    assert True