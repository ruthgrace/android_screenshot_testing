"""
Android Accessibility Testing Library

A Python library for testing Android apps using ADB commands, screenshots,
and Claude LLM for visual assertions.
"""

import os
import subprocess
import base64
import json
import time
from typing import Optional, List
from anthropic import Anthropic
from anthropic import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError


class ValidationResult:
    """Result of a screenshot validation."""

    def __init__(self, result: bool, error: Optional[str] = None):
        """
        Initialize validation result.

        Args:
            result: Whether the validation passed
            error: Error message if validation failed, None otherwise
        """
        self.result = result
        self.error = error

    def __bool__(self):
        """Allow using the result directly in assertions."""
        return self.result

    def __repr__(self):
        """String representation of the result."""
        if self.result:
            return "ValidationResult(result=True)"
        return f"ValidationResult(result=False, error={self.error!r})"


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

    def validate_screenshot(self, screenshot_path: str, description: str,
                           model: Optional[str] = None) -> ValidationResult:
        """
        Validate that a screenshot matches the given description using Claude LLM.

        Args:
            screenshot_path: Path to the screenshot file
            description: Description of what should be visible in the screenshot
            model: Claude model to use for validation. If None, uses the default model from initialization.

        Returns:
            ValidationResult object with .result (bool) and .error (str or None)
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

Based on the screenshot, does it match this description?
Respond with ONLY a JSON object in this exact format:
{{"result": true, "error_message": null}}
or
{{"result": false, "error_message": "description of what is wrong or missing"}}

If ALL the key elements described are present and the description is completely accurate, set result to true and error_message to null.
If any key elements are missing or the description doesn't match, set result to false and provide a clear error_message explaining what is wrong."""

        # Call Claude API with exponential backoff retry
        retry_delays = [1, 2, 4, 8]  # Exponential backoff in seconds
        last_error = None

        for attempt, delay in enumerate(retry_delays + [None]):  # +[None] for one final attempt
            try:
                message = self.client.messages.create(
                    model=model,
                    max_tokens=200,
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
                response_text = message.content[0].text.strip()
                response_data = json.loads(response_text)
                return ValidationResult(
                    result=response_data["result"],
                    error=response_data.get("error_message")
                )

            except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                # If this was the last attempt, raise the error
                if delay is None:
                    break

                # Log retry attempt
                print(f"API error on attempt {attempt + 1}: {type(e).__name__}. Retrying in {delay}s...")
                time.sleep(delay)

        # If we exhausted all retries, raise the last error
        raise last_error

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
        # Also escape special shell characters
        escaped_text = text.replace(" ", "%s")
        escaped_text = escaped_text.replace("(", "\\(")
        escaped_text = escaped_text.replace(")", "\\)")
        self.shell(f"input text {escaped_text}")

    def press_key(self, keycode: str):
        """
        Press a key by keycode.

        Args:
            keycode: Android keycode (e.g., "KEYCODE_BACK", "KEYCODE_HOME")
        """
        self.shell(f"input keyevent {keycode}")

    def open_app(self, package_name: str):
        """
        Open an app by package name.

        Args:
            package_name: Package name of the app (e.g., "com.android.settings")
        """
        self.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")

    def press_back(self):
        """Press the Android back button."""
        self.press_key("KEYCODE_BACK")