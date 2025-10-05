# android_screenshot_testing

Accessible apps are notoriously hard to test on Android. This is a Python framework for integration testing accessibility services with ADB and screenshots. Importantly, it does not itself use accessibility services so it does not conflict with an accessibilty service app.

# Example

You can set up the example like so:

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

Set your Anthropic API key and run the example:

```
export ANTHROPIC_API_KEY=your_api_key_here
./venv/bin/pytest example_test.py
```

I recommend using droidrun to identify the coordinates of elements you need to click on the screen (though Droidrun cannot be running while testing an app that uses accessibility services since only one app can hold the permissions at a time)
