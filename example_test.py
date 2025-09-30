"""
Example usage of the Android Accessibility Tester library.
"""

from android_accessibility_tester import AndroidAccessibilityTester


def main():
    # Initialize the tester
    tester = AndroidAccessibilityTester()

    # List connected devices
    devices = tester.list_devices()
    print(f"Connected devices: {devices}")

    # Execute a shell command
    print(f"Device model: {tester.shell('getprop ro.product.model')}")

    # Take a screenshot
    screenshot_path = "test_screenshot.png"
    tester.screenshot(screenshot_path)
    print(f"Screenshot saved to: {screenshot_path}")

    # Assert screenshot matches description
    description = "The home screen with app icons visible"
    matches = tester.assert_screenshot(screenshot_path, description)
    print(f"Screenshot matches description: {matches}")

    # Example interactions
    # tester.tap(500, 1000)  # Tap at coordinates
    # tester.swipe(500, 1500, 500, 500)  # Swipe up
    # tester.input_text("Hello World")  # Type text
    # tester.press_key("KEYCODE_BACK")  # Press back button


if __name__ == "__main__":
    main()