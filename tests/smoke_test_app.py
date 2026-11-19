"""Headless smoke test: load the app, predict each example customer, fail on any exception.

Run from the repository root:  python tests/smoke_test_app.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py"

app_test = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
assert not app_test.exception, f"Exception on load: {app_test.exception}"

start_from = next(selectbox for selectbox in app_test.selectbox if selectbox.label == "Partir de")
for example_option in start_from.options:
    start_from.set_value(example_option).run()
    next(button for button in app_test.button if button.label == "Predecir riesgo de abandono").click().run()
    assert not app_test.exception, f"Exception after predicting '{example_option}': {app_test.exception}"
    shown_metrics = {metric.label: metric.value for metric in app_test.metric}
    print(f"{example_option}: churn probability {shown_metrics.get('Probabilidad de abandono')} | {shown_metrics.get('Segmento de riesgo')}")

    # What-if panel: simulate a two-year contract (reruns only the fragment; the prediction must stay on screen)
    next(selectbox for selectbox in app_test.selectbox if selectbox.label == "Tipo de contrato (simulado)").set_value("Two year").run()
    assert not app_test.exception, f"Exception in what-if for '{example_option}': {app_test.exception}"
    shown_metrics = {metric.label: metric.value for metric in app_test.metric}
    money_now, money_after = (float(shown_metrics[label].strip("$/mes").replace(",", ""))
                              for label in ["Dinero en riesgo hoy", "Dinero en riesgo tras el cambio"])
    print(f"  what-if two-year contract: risk {shown_metrics['Riesgo tras el cambio']} | "
          f"money at risk ${money_now:,.2f} -> ${money_after:,.2f}/month")
    if example_option == "Ejemplo: cliente de prueba con mayor riesgo":
        assert money_after < money_now, "A two-year contract should lower the money at risk of the riskiest customer"
    start_from = next(selectbox for selectbox in app_test.selectbox if selectbox.label == "Partir de")

print("SMOKE TEST PASSED")
