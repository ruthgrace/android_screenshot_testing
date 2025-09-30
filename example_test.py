#!/usr/bin/env python3

"""
Example usage of the Android Accessibility Tester library.
"""
import android_accessibility_tester
import subprocess
import os
import pytest


def install_debug_app(force=False):
    """Install the debug app, optionally with force restart."""
    script_path = os.path.join(os.path.dirname(__file__), 'install.sh')
    cmd = [script_path]
    if force:
        cmd.append('--force')
    subprocess.run(cmd, check=True)


@pytest.fixture(scope="session")
def app_installed():
    """Install the debug app once for all tests."""
    install_debug_app(force=True)
    yield


@pytest.fixture(scope="function")
def tester(app_installed):
    """Create a fresh tester instance and open the app for each test."""
    tester = android_accessibility_tester.AndroidAccessibilityTester()
    tester.open_app("com.example.whiz.debug")
    yield tester
    # Add any per-test cleanup here if needed


# Example test - add your actual tests below
def test_app_open():
    """Test that Clock app opens with the correct tabs."""
    tester = android_accessibility_tester.AndroidAccessibilityTester()

    # Open the Clock app
    tester.open_app("com.google.android.deskclock")

    # Take a screenshot
    os.makedirs("screenshots", exist_ok=True)
    screenshot_path = "screenshots/clock_app_screenshot.png"
    tester.screenshot(screenshot_path)

    # Assert that all three tabs are visible at the bottom
    assert tester.validate_screenshot(
        screenshot_path,
        'The screen shows a clock application with tabs at the bottom. The tabs should include "Alarms", "Timer", and "Stopwatch".'
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
