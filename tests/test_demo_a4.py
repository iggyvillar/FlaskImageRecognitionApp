"""
Assignment 4 - Demonstration Test File
NOW PASSING - Shows full pipeline execution
"""

import pytest


def test_pipeline_stops_on_failure():
    """
    AC: Demonstrate full pipeline execution when all tests pass
    
    Given: All tests are passing
    When: The pipeline runs
    Then: All 4 jobs should execute successfully in order
    """
    # FIXED - Now passes
    assert True, "✅ Test fixed - full pipeline should execute"


def test_this_one_passes():
    """
    AC: This test demonstrates success
    """
    assert True