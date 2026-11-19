"""Headless smoke test: load the app, predict each example customer, fail on any exception.

Run from the repository root:  python tests/smoke_test_app.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py"

app_test = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
assert not app_test.exception, f"Exception on load: {app_test.exception}"

start_from = next(selectbox for selectbox in app_test.selectbox if selectbox.label == "Start from")
for example_option in start_from.options:
    start_from.set_value(example_option).run()
    next(button for button in app_test.button if button.label == "Predict churn risk").click().run()
    assert not app_test.exception, f"Exception after predicting '{example_option}': {app_test.exception}"
    shown_metrics = {metric.label: metric.value for metric in app_test.metric}
    print(f"{example_option}: churn probability {shown_metrics.get('Churn probability')} | {shown_metrics.get('Risk segment')}")
    start_from = next(selectbox for selectbox in app_test.selectbox if selectbox.label == "Start from")

print("SMOKE TEST PASSED")
