"""
Android Accessibility Testing Library

A Python library for testing Android apps using ADB commands, screenshots,
and Claude LLM for visual assertions.
"""

import os
import subprocess
import base64
from typing import Optional, List
from anthropic import Anthropic


class AndroidAccessibilityTester:
    """Main class for Android accessibility testing with screenshot assertions."""

    def __init__(self, device_id: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the tester.

        Args:
            device_id: Specific Android device ID. If None, uses the first available device.
            model: Claude model to use for assertions. If None, defaults to "claude-sonnet-4-20250514".
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable.
        """
        self.device_id = device_id
        self.default_model = model or "claude-sonnet-4-20250514"
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._verify_adb()

    def _verify_adb(self):
        """Verify ADB is installed and accessible."""
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ADB not found. Please install Android SDK Platform Tools.")

    def _get_adb_command(self) -> List[str]:
        """Get base ADB command with device specification."""
        cmd = ["adb"]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        return cmd

    def shell(self, command: str) -> str:
        """
        Execute an ADB shell command.

        Args:
            command: Shell command to execute

        Returns:
            Command output as string
        """
        cmd = self._get_adb_command() + ["shell", command]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def screenshot(self, output_path: str) -> str:
        """
        Capture a screenshot from the Android device.

        Args:
            output_path: Local path to save the screenshot

        Returns:
            Path to the saved screenshot
        """
        # Take screenshot on device
        device_path = "/sdcard/screenshot.png"
        self.shell(f"screencap -p {device_path}")

        # Pull screenshot to local machine
        cmd = self._get_adb_command() + ["pull", device_path, output_path]
        subprocess.run(cmd, capture_output=True, check=True)

        # Clean up device screenshot
        self.shell(f"rm {device_path}")

        return output_path

    def assert_screenshot(self, screenshot_path: str, description: str,
                         model: Optional[str] = None) -> bool:
        """
        Assert that a screenshot matches the given description using Claude LLM.

        Args:
            screenshot_path: Path to the screenshot file
            description: Description of what should be visible in the screenshot
            model: Claude model to use for assertion. If None, uses the default model from initialization.

        Returns:
            True if screenshot matches description, False otherwise
        """
        if model is None:
            model = self.default_model
        # Read and encode the screenshot
        with open(screenshot_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Create prompt for Claude
        prompt = f"""You are analyzing a screenshot from an Android application accessibility test.

Description of what should be in the screenshot:
{description}

Based on the screenshot, does it match this description? Respond with ONLY "true" or "false" (lowercase).
If the key elements described are present and the description is accurate, respond "true".
If any key elements are missing or the description doesn't match, respond "false"."""

        # Call Claude API
        message = self.client.messages.create(
            model=model,
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        # Parse response
        response_text = message.content[0].text.strip().lower()
        return response_text == "true"

    def list_devices(self) -> List[str]:
        """
        List all connected Android devices.

        Returns:
            List of device IDs
        """
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        devices = [line.split()[0] for line in lines if line.strip() and "device" in line]
        return devices

    def tap(self, x: int, y: int):
        """
        Simulate a tap at the given coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300):
        """
        Simulate a swipe gesture.

        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            duration_ms: Duration of swipe in milliseconds
        """
        self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str):
        """
        Input text into the focused field.

        Args:
            text: Text to input (spaces will be replaced with %s)
        """
        # ADB shell input text requires spaces to be encoded
        escaped_text = text.replace(" ", "%s")
        self.shell(f"input text {escaped_text}")

    def press_key(self, keycode: str):
        """
        Press a key by keycode.

        Args:
            keycode: Android keycode (e.g., "KEYCODE_BACK", "KEYCODE_HOME")
        """
        self.shell(f"input keyevent {keycode}")