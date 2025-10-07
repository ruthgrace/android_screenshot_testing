#!/usr/bin/env python3

"""
Test script for find_element_coordinates_accessibility method.
"""

import android_accessibility_tester

def test_find_clock_app():
    """Test finding the Clock app icon coordinates."""
    tester = android_accessibility_tester.AndroidAccessibilityTester()

    # Find the Clock app
    coords = tester.find_element_coordinates_accessibility("Clock app icon")

    if coords:
        print(f"Found Clock app!")
        print(f"Coordinates: ({coords.x1}, {coords.y1}) to ({coords.x2}, {coords.y2})")
        print(f"Center: {coords.center()}")
        print(f"Method: {coords.method}")
        print(f"Explanation: {coords.confidence}")

        # Optionally tap it to verify
        center_x, center_y = coords.center()
        print(f"\nTapping at center ({center_x}, {center_y})...")
        tester.tap(center_x, center_y)
    else:
        print("Clock app not found!")


def test_find_chrome_app():
    """Test finding the Chrome app icon coordinates."""
    tester = android_accessibility_tester.AndroidAccessibilityTester()

    # Go to home screen first
    tester.press_key("KEYCODE_HOME")

    # Find the Chrome app
    coords = tester.find_element_coordinates_accessibility("Chrome browser app")

    if coords:
        print(f"Found Chrome app!")
        print(f"Coordinates: ({coords.x1}, {coords.y1}) to ({coords.x2}, {coords.y2})")
        print(f"Center: {coords.center()}")
        print(f"Method: {coords.method}")
        print(f"Explanation: {coords.confidence}")
    else:
        print("Chrome app not found!")


if __name__ == "__main__":
    print("Testing find_element_coordinates_accessibility...")
    print("\n=== Test 1: Find Clock app ===")
    test_find_clock_app()

    print("\n=== Test 2: Find Chrome app ===")
    test_find_chrome_app()
