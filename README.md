# Menolyte Core

Churn prediction and retention prioritization for small ISPs: **who** is likely to leave, **why**, and **what to do**
about it. Final project for the Machine Learning course (Yachay Tech).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Smoke test (loads the app and predicts the example customers without opening a browser):

```bash
python tests/smoke_test_app.py
```

## Contents

| Path | Description |
|---|---|
| `streamlit_app/app.py` | Streamlit app: EDA, model performance, prediction & explainability, business view. |
| `models/` | Exported SVM pipeline and results (the app never retrains). |
| `data/` | Telco Customer Churn dataset (7,043 customers). |
| `telco_churn_optimization_xai.ipynb` | Modeling notebook: pipelines, Grid/Random/TPE, NSGA-II, SHAP and LIME. |

## Use of Artificial Intelligence

The code was generated with the assistance of the Claude agent (Claude Code), used as a tool to generate code and to
understand concepts and results. All technical and business decisions belong to the student.
