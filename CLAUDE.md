# Android Accessibility Tester

A Python framework for integration testing Android apps that use accessibility services, using ADB commands, screenshots, and Claude AI for visual validation.

## Overview

This framework enables testing Android accessibility services without requiring another accessibility service to run the tests (which would conflict). It uses ADB shell commands to interact with devices and Claude's vision capabilities to validate UI states through screenshots.

## Key Features

- **ADB-based interaction**: Control Android devices via shell commands (tap, swipe, input text, etc.)
- **AI-powered visual assertions**: Use Claude to validate screenshots against natural language descriptions
- **Accessibility hierarchy analysis**: Find elements using UI accessibility tree with LLM assistance
- **Exponential backoff retry**: Automatic retry logic for Claude API calls with transient errors
- **pytest integration**: Built as pytest fixtures for easy test authoring

## Core Components

### AndroidAccessibilityTester

Main class located in `android_accessibility_tester.py:75`

**Initialization:**
```python
tester = AndroidAccessibilityTester(
    device_id=None,      # Specific device ID, or None for first available
    model=None,          # Claude model, defaults to "claude-sonnet-4-20250514"
    api_key=None         # Anthropic API key, defaults to ANTHROPIC_API_KEY env var
)
```

### Device Control Methods

- `shell(command)` - Execute arbitrary ADB shell commands (`android_accessibility_tester.py:106`)
- `screenshot(output_path)` - Capture device screenshot (`android_accessibility_tester.py:120`)
- `tap(x, y)` - Simulate tap at coordinates (`android_accessibility_tester.py:240`)
- `swipe(x1, y1, x2, y2, duration_ms=300)` - Swipe gesture (`android_accessibility_tester.py:250`)
- `input_text(text)` - Type text into focused field (`android_accessibility_tester.py:263`)
- `press_key(keycode)` - Press Android keycode (`android_accessibility_tester.py:277`)
- `press_back()` - Press back button (`android_accessibility_tester.py:300`)
- `open_app(package_name)` - Launch app by package name (`android_accessibility_tester.py:286`)
- `list_devices()` - List connected Android devices (`android_accessibility_tester.py:228`)

### Pixel Change Detection (New in v2.0)

**wait_for_pixel_change(x, y, timeout=10.0, poll_interval=0.5)**

Wait for a pixel at given coordinates to change color. This is a fast, lightweight method for detecting UI changes without requiring accessibility services or UI hierarchy parsing (`android_accessibility_tester.py:333`).

Arguments:
- `x` - X coordinate of pixel to monitor
- `y` - Y coordinate of pixel to monitor
- `timeout` - Maximum wait time in seconds (default: 10.0)
- `poll_interval` - Time between checks in seconds (default: 0.5)

Returns dictionary with:
- `'changed'` - Boolean, whether pixel changed
- `'initial_color'` - Hex color code of initial pixel (e.g., '#FF0000')
- `'final_color'` - Hex color code after change (if changed)
- `'error'` - Error message if timeout occurred

Example:
```python
# Wait for a loading indicator to appear/disappear
result = tester.wait_for_pixel_change(500, 300, timeout=5.0)
if result['changed']:
    print(f"Color changed from {result['initial_color']} to {result['final_color']}")
else:
    print(f"Timeout: {result['error']}")

# Use in assertions
result = tester.wait_for_pixel_change(100, 200, timeout=3.0)
assert result['changed'], result['error']
```

**Use cases:**
- Detect loading spinners appearing/disappearing
- Wait for buttons to enable/disable (color change)
- Detect animations starting/stopping
- Monitor progress indicators

**Note:** Requires Pillow library (automatically installed with v2.0+)

### Visual Validation

**validate_screenshot(screenshot_path, description, model=None)**

Uses Claude to validate that a screenshot matches a natural language description (`android_accessibility_tester.py:143`).

Returns `ValidationResult` object:
- `.result` - Boolean indicating if validation passed
- `.error` - Error message if validation failed, None otherwise
- Castable to boolean for use in assertions

Example:
```python
result = tester.validate_screenshot(
    "screenshots/screen.png",
    'The screen shows a clock app with tabs at the bottom including "Alarms", "World Clock", "Timers", "Stopwatch", and "Bedtime".'
)
assert result  # Will show result.error in pytest output if it fails
```

**Important:** The description should be precise and comprehensive. The validation only passes if ALL key elements described are present and accurate (`android_accessibility_tester.py:174`).

### Element Waiting (New in v2.0)

**wait_for_element(text=None, resource_id=None, content_desc=None, timeout=10.0, poll_interval=0.5)**

Wait for an element to appear on screen. Uses `uiautomator dump` to poll the UI hierarchy without requiring accessibility services, avoiding conflicts with apps that hold accessibility permissions (`android_accessibility_tester.py:495`).

Arguments:
- `text` - Text to match (optional)
- `resource_id` - Resource ID to match, can be short form like "button_id" or full form (optional)
- `content_desc` - Content description to match (optional)
- `timeout` - Maximum wait time in seconds (default: 10.0)
- `poll_interval` - Time between polls in seconds (default: 0.5)

Returns: `True` if element appears within timeout, `False` otherwise

Example:
```python
# Wait for a button by resource ID
if tester.wait_for_element(resource_id="start_button", timeout=5.0):
    print("Button appeared!")

# Wait for specific text
assert tester.wait_for_element(text="Welcome", timeout=3.0)
```

**wait_until_gone(text=None, resource_id=None, content_desc=None, timeout=10.0, poll_interval=0.5)**

Wait for an element to disappear from screen. Uses the same non-intrusive polling mechanism as `wait_for_element` (`android_accessibility_tester.py:533`).

Arguments: Same as `wait_for_element`

Returns: `True` if element disappears within timeout, `False` otherwise

Example:
```python
# Wait for loading spinner to disappear
tester.wait_until_gone(resource_id="loading_spinner", timeout=30.0)

# Wait for text to disappear
assert tester.wait_until_gone(text="Loading...", timeout=5.0)
```

**Important:** At least one of `text`, `resource_id`, or `content_desc` must be provided, or `ValueError` will be raised.

### Element Location

**find_element_coordinates_accessibility(description, model=None)**

Uses Claude to analyze the accessibility hierarchy and find element coordinates (`android_accessibility_tester.py:372`).

Returns `ElementCoordinates` object:
- `.x1, .y1` - Top-left coordinates
- `.x2, .y2` - Bottom-right coordinates
- `.center()` - Returns center point tuple `(x, y)`
- `.method` - Method used ("accessibility" or "visual")
- `.confidence` - Explanation/confidence from the method

Example:
```python
coords = tester.find_element_coordinates_accessibility("the Start Timer button")
if coords:
    x, y = coords.center()
    tester.tap(x, y)
```

## Setup & Usage

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Prerequisites

- **ADB**: Android SDK Platform Tools must be installed
- **Anthropic API Key**: Set `ANTHROPIC_API_KEY` environment variable
- **Connected device**: Android device connected via USB or emulator running

### Running Tests

```bash
export ANTHROPIC_API_KEY=your_api_key_here
./venv/bin/pytest example_test.py
```

## Example Test Structure

See `example_test.py` for complete examples. Key patterns:

```python
import android_accessibility_tester
import pytest
import os

@pytest.fixture(scope="function")
def tester():
    """Create tester and open app for each test."""
    tester = android_accessibility_tester.AndroidAccessibilityTester()
    tester.open_app("com.example.myapp")
    yield tester

def test_my_feature(tester):
    """Test a specific feature."""
    os.makedirs("screenshots", exist_ok=True)

    # Interact with the app
    tester.tap(500, 300)

    # Capture and validate
    screenshot_path = "screenshots/feature.png"
    tester.screenshot(screenshot_path)

    assert tester.validate_screenshot(
        screenshot_path,
        "Description of expected UI state"
    )
```

## Error Handling

### Transient API Errors

The framework automatically retries Claude API calls with exponential backoff (1s, 2s, 4s, 8s) for:
- `RateLimitError`
- `InternalServerError`
- `APITimeoutError`
- `APIConnectionError`

Implementation: `android_accessibility_tester.py:178` and `android_accessibility_tester.py:410`

### Validation Failures

When `validate_screenshot()` returns False, the `.error` field contains Claude's explanation of what's missing or incorrect. This appears in pytest output when assertions fail.

## Implementation Notes

### Text Input Encoding

The `input_text()` method automatically handles ADB text input requirements:
- Spaces replaced with `%s`
- Special characters like `()` are escaped
- Implementation: `android_accessibility_tester.py:272`

### UI Hierarchy Simplification

The accessibility hierarchy is simplified before sending to Claude to reduce token usage:
- Only includes meaningful attributes (text, content_desc, resource_id, bounds, clickable, enabled)
- Truncates package names to just class names
- Filters out children without meaningful content
- Max depth of 5 levels
- Implementation: `android_accessibility_tester.py:337`

### Screenshot Workflow

1. Device captures screenshot to `/sdcard/screenshot.png`
2. ADB pulls file to local path
3. Device file is cleaned up
4. Implementation: `android_accessibility_tester.py:130`

## Tips & Best Practices

1. **Use Droidrun for coordinates**: The README recommends using [Droidrun](https://github.com/anthropics/droidrun) to identify element coordinates (note: Droidrun cannot run simultaneously with accessibility services)

2. **Precise descriptions**: Make screenshot validation descriptions as specific as possible - validation only passes if all elements match exactly

3. **Screenshots directory**: Create `screenshots/` directory for test artifacts

4. **Per-test cleanup**: Use pytest fixtures with function scope to ensure clean state between tests

5. **Model selection**: Default model is `claude-sonnet-4-20250514`, but you can override per-call or at initialization

## Architecture

The framework is designed to avoid conflicts with accessibility service apps:
- Uses ADB shell commands instead of accessibility APIs
- Runs externally to the Android system
- Can test apps that hold accessibility service permissions
- Ideal for integration testing accessibility features

## License

See LICENSE.md
