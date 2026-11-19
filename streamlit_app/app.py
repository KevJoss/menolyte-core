"""Menolyte Core — churn risk platform for mid-sized ISPs.

Loads the pipeline and results exported by telco_churn_optimization_xai.ipynb. Nothing is retrained here.
Run locally from the project folder:  streamlit run streamlit_app/app.py
"""
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import fbeta_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find styles.py also when run from tests (AppTest)
from styles import (action_card, apply_global_styles, brand_header, figure_card, html, register_plotly_template,
                    risk_badge, risk_scale, step_card)

warnings.filterwarnings("ignore")

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIRECTORY / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIRECTORY = PROJECT_DIRECTORY / "models"

HIGHLIGHT_COLOR = "#d62728"
NEUTRAL_COLOR = "#b0b7c3"
RETENTION_COLOR = NEUTRAL_COLOR  # "lowers risk" in brand grey: only risk is colored (red)
RISK_SEGMENT_NAMES = ["Riesgo bajo", "Riesgo medio", "Riesgo alto"]
HIGH_RISK_NAME = RISK_SEGMENT_NAMES[-1]

st.set_page_config(page_title="Menolyte Core", page_icon="📡", layout="wide")
apply_global_styles()
register_plotly_template()


# ----------------------------------------------------------------------------------------------------------------------
# Loading (cached: the model and results are read once per session)
# ----------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_model_and_results():
    final_pipeline = joblib.load(MODELS_DIRECTORY / "final_svm_pipeline.joblib")
    app_results = joblib.load(MODELS_DIRECTORY / "app_results.joblib")
    return final_pipeline, app_results


@st.cache_data
def load_customer_data():
    customer_data = pd.read_csv(DATA_PATH)
    customer_data["TotalCharges"] = pd.to_numeric(customer_data["TotalCharges"].str.strip().replace("", "0"))
    customer_data["churned"] = customer_data["Churn"].map({"Yes": 1, "No": 0})
    return customer_data


final_pipeline, app_results = load_model_and_results()
customer_data = load_customer_data()
metadata = app_results["metadata"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
category_values = metadata["category_values"]
feature_columns = list(app_results["features_train"].columns)
risk_segment_edges = metadata["risk_segment_edges"]

fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
fitted_svm = final_pipeline.named_steps["model"]
transformed_feature_names = fitted_preprocessor.get_feature_names_out()


def original_variable_of(transformed_name):
    without_prefix = transformed_name.split("__", 1)[1]
    return next(column for column in feature_columns if without_prefix == column or without_prefix.startswith(column + "_"))


transformed_to_original = [original_variable_of(name) for name in transformed_feature_names]


def risk_segment_of(probability):
    for segment_name, upper_edge in zip(RISK_SEGMENT_NAMES, risk_segment_edges[1:]):
        if probability < upper_edge:
            return segment_name
    return RISK_SEGMENT_NAMES[-1]


@st.cache_resource
def build_shap_explainer():
    return shap.KernelExplainer(lambda transformed_rows: fitted_svm.predict_proba(transformed_rows)[:, 1],
                                app_results["shap_background"])


def encode_for_lime(original_rows):
    encoded_rows = original_rows[feature_columns].copy()
    for column in categorical_features:
        encoded_rows[column] = encoded_rows[column].map({value: code for code, value in enumerate(category_values[column])})
    return encoded_rows.to_numpy(dtype=float)


def churn_probability_from_lime_rows(encoded_rows):
    decoded_rows = pd.DataFrame(encoded_rows, columns=feature_columns)
    for column in categorical_features:
        decoded_rows[column] = [category_values[column][int(round(code))] for code in decoded_rows[column]]
    for column in numeric_features:
        decoded_rows[column] = decoded_rows[column].astype(float)
    return final_pipeline.predict_proba(decoded_rows)


@st.cache_resource
def build_lime_explainer():
    categorical_indices = [feature_columns.index(column) for column in categorical_features]
    return LimeTabularExplainer(
        training_data=encode_for_lime(app_results["features_train"]),
        feature_names=feature_columns,
        categorical_features=categorical_indices,
        categorical_names={index: category_values[column] for index, column in zip(categorical_indices, categorical_features)},
        class_names=["Stays", "Churns"],
        discretize_continuous=True,
        mode="classification",
        random_state=42,
    )


PLAIN_LANGUAGE_LABELS = {
    "tenure": "Antigüedad", "Contract": "Tipo de contrato", "MonthlyCharges": "Cuota mensual",
    "TotalCharges": "Total pagado", "PaymentMethod": "Método de pago", "OnlineSecurity": "Seguridad en línea",
    "TechSupport": "Soporte técnico", "InternetService": "Tipo de internet", "SeniorCitizen": "Adulto mayor",
    "PaperlessBilling": "Factura electrónica", "OnlineBackup": "Respaldo en línea", "DeviceProtection": "Protección de equipos",
    "StreamingTV": "TV por streaming", "StreamingMovies": "Películas por streaming", "MultipleLines": "Varias líneas",
    "PhoneService": "Servicio telefónico", "Dependents": "Dependientes", "Partner": "Pareja", "gender": "Género",
}

# Display only: the data and the model keep the original (English) category values
CATEGORY_DISPLAY_NAMES = {
    "Month-to-month": "Mes a mes", "One year": "Un año", "Two year": "Dos años",
    "DSL": "DSL", "Fiber optic": "Fibra óptica",
    "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)", "Credit card (automatic)": "Tarjeta de crédito (automática)",
    "Yes": "Sí", "No": "No", "No internet service": "Sin servicio de internet", "No phone service": "Sin servicio telefónico",
    "Male": "Hombre", "Female": "Mujer", "0": "No", "1": "Sí",
}


def display_value(value):
    """Spanish name of a category value, for display only."""
    return CATEGORY_DISPLAY_NAMES.get(str(value), str(value))


def describe_lime_condition(condition):
    """LIME condition in plain Spanish, e.g. 'Contract=Month-to-month' -> 'Tipo de contrato: Mes a mes'."""
    categorical_match = re.fullmatch(r"(\w+)=(.+)", condition)
    if categorical_match:
        variable, value = categorical_match.groups()
        return f"{PLAIN_LANGUAGE_LABELS.get(variable, variable)}: {display_value(value)}"
    variable_pattern = r"\b(" + "|".join(sorted(PLAIN_LANGUAGE_LABELS, key=len, reverse=True)) + r")\b"
    return re.sub(variable_pattern, lambda match: PLAIN_LANGUAGE_LABELS[match.group(1)], condition)


def plain_probability_sentence(probability):
    """Churn probability as natural frequency ('7 of 10'); switches to 'of 100' for low risks to avoid '0 of 10'."""
    out_of_ten, out_of_hundred = round(probability * 10), round(probability * 100)
    if out_of_ten >= 1:
        return f"Según el modelo, de cada 10 clientes con este perfil, unos <b>{out_of_ten}</b> se irían."
    if out_of_hundred >= 1:
        return (f"Según el modelo, de cada 100 clientes con este perfil, solo <b>{out_of_hundred}</b> "
                f"se {'iría' if out_of_hundred == 1 else 'irían'}.")
    return "Según el modelo, casi ningún cliente con este perfil se iría."


def describe_value(variable, value):
    if variable == "tenure":
        return f"{int(value)} mes" + ("" if int(value) == 1 else "es")
    if variable in ["MonthlyCharges", "TotalCharges"]:
        return f"${value:,.0f}"
    if variable == "SeniorCitizen":
        return "Sí" if value == 1 else "No"
    return display_value(value)


def suggested_action(variable, value):
    """Retention action for a risk driver (hypotheses to validate with an A/B test, not causal guarantees)."""
    if variable == "Contract" and value == "Month-to-month":
        return "Ofrecer un contrato de uno o dos años con descuento."
    if variable == "tenure" and value <= 12:
        return "Incluirlo en el programa de primeros meses (llamada de bienvenida, revisión de uso)."
    if variable == "MonthlyCharges" and value >= 70:
        return "Revisar el plan: sugerir uno más barato que se ajuste a su uso real, o un descuento por fidelidad."
    if variable in ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"] and value == "No":
        return f"Ofrecer una prueba gratuita de {PLAIN_LANGUAGE_LABELS[variable].lower()} por unos meses."
    if variable == "PaymentMethod" and value == "Electronic check":
        return "Ofrecer un pequeño crédito en la factura por pasarse al pago automático."
    if variable == "InternetService" and value == "Fiber optic":
        return "Revisar la calidad del servicio de fibra y si su precio es competitivo para este cliente."
    return None


@st.cache_data(show_spinner=False)
def explain_customer(customer_values):
    """SHAP per original variable + LIME conditions. Cached so reruns (what-if, other tabs) don't recompute them."""
    customer_row = pd.DataFrame([customer_values])[feature_columns]
    transformed_customer = fitted_preprocessor.transform(customer_row)
    transformed_shap_values = build_shap_explainer().shap_values(transformed_customer, nsamples=300, silent=True)[0]
    shap_by_variable = pd.Series(transformed_shap_values, index=transformed_feature_names).groupby(transformed_to_original).sum()
    lime_explanation = build_lime_explainer().explain_instance(
        encode_for_lime(customer_row)[0], churn_probability_from_lime_rows, num_features=8, num_samples=1500, labels=(1,))
    return shap_by_variable, lime_explanation.as_list(label=1)


# Short form: only the most important variables are asked (SHAP ranking of the SVM, computed on training data).
# TotalCharges is not asked: it is almost exactly tenure × MonthlyCharges, so it is computed.
FIELD_RANKING = [variable for variable in app_results["importance_comparison"].sort_values("SHAP rank").index
                 if variable != "TotalCharges"]
FORM_FIELD_GROUPS = {"Cuenta": ["Contract", "tenure", "MonthlyCharges", "PaymentMethod"],
                     "Servicios de internet": ["InternetService", "OnlineSecurity", "TechSupport"]}
FORM_FIELDS = [variable for group_variables in FORM_FIELD_GROUPS.values() for variable in group_variables]
assert set(FORM_FIELDS) == set(FIELD_RANKING[:len(FORM_FIELDS)]), "The form must ask the top-ranked variables"
INTERNET_ADD_ONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
typical_customer_values = {column: (app_results["features_train"][column].median() if column in numeric_features
                                    else app_results["features_train"][column].mode()[0]) for column in feature_columns}


def complete_customers(asked_rows, asked_fields):
    """Fill the fields not asked with the typical training value, keep services consistent and compute TotalCharges."""
    completed_rows = pd.DataFrame({column: asked_rows[column] if column in asked_fields else typical_customer_values[column]
                                   for column in feature_columns}, index=asked_rows.index)
    no_internet = completed_rows["InternetService"] == "No"
    for add_on in INTERNET_ADD_ONS:
        completed_rows[add_on] = np.where(no_internet, "No internet service",
                                          completed_rows[add_on].replace("No internet service", "No"))
    completed_rows["MultipleLines"] = np.where(completed_rows["PhoneService"] == "No", "No phone service",
                                               completed_rows["MultipleLines"].replace("No phone service", "No"))
    completed_rows["TotalCharges"] = completed_rows["tenure"] * completed_rows["MonthlyCharges"]
    return completed_rows


@st.cache_data(show_spinner="Midiendo cuántos datos del formulario necesita el modelo...")
def form_size_curve():
    """Test-set scores when only the top-k ranked fields are asked (the rest filled in). The model is not retrained."""
    test_predictions = app_results["test_predictions"]
    test_customers = customer_data.loc[test_predictions["customer_index"].astype(int), feature_columns].reset_index(drop=True)
    actual_churn = test_predictions["actual_churn"].to_numpy()
    all_fields_decisions = final_pipeline.predict(test_customers)
    curve_rows = []
    for field_count in range(1, len(FIELD_RANKING) + 1):
        completed_customers = complete_customers(test_customers, FIELD_RANKING[:field_count])
        decisions = final_pipeline.predict(completed_customers)
        curve_rows.append({"fields asked": field_count, "field added": FIELD_RANKING[field_count - 1],
                           "F2": fbeta_score(actual_churn, decisions, beta=2),
                           "ROC-AUC": roc_auc_score(actual_churn, final_pipeline.predict_proba(completed_customers)[:, 1]),
                           "same decision as all fields": (decisions == all_fields_decisions).mean()})
    return pd.DataFrame(curve_rows)


AUTOMATIC_PAYMENT_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]


def actionable_single_changes(customer_values):
    """Retention levers the ISP controls, one variable at a time: longer contract, add support/security, auto-pay."""
    changes = []
    contract_order = category_values["Contract"]
    for longer_contract in contract_order[contract_order.index(customer_values["Contract"]) + 1:]:
        changes.append(("Contract", longer_contract))
    if customer_values["InternetService"] != "No":
        changes += [(variable, "Yes") for variable in ["TechSupport", "OnlineSecurity"] if customer_values[variable] == "No"]
    if customer_values["PaymentMethod"] not in AUTOMATIC_PAYMENT_METHODS:
        changes += [("PaymentMethod", method) for method in AUTOMATIC_PAYMENT_METHODS]
    return changes


def churn_probabilities_for(customer_rows):
    return final_pipeline.predict_proba(pd.DataFrame(customer_rows)[feature_columns])[:, 1]


@st.fragment  # widgets inside rerun only this function: the prediction stays on screen and SHAP/LIME are not recomputed
def show_what_if_panel(customer_values, current_probability):
    st.markdown("#### ¿Y si…? Simula una acción de retención")
    monthly_fee = customer_values["MonthlyCharges"]
    money_at_risk_now = current_probability * monthly_fee

    candidate_changes = actionable_single_changes(customer_values)
    if candidate_changes:
        candidate_probabilities = churn_probabilities_for([{**customer_values, variable: new_value}
                                                           for variable, new_value in candidate_changes])
        best_index = int(np.argmin(candidate_probabilities))
        best_variable, best_value = candidate_changes[best_index]
        best_probability = candidate_probabilities[best_index]
    if candidate_changes and best_probability < current_probability:
        st.success(f"**Acción recomendada:** cambiar *{PLAIN_LANGUAGE_LABELS[best_variable].lower()}* a "
                   f"**{display_value(best_value)}**. Riesgo de abandono {current_probability:.0%} → {best_probability:.0%}, "
                   f"protege ~**${(current_probability - best_probability) * monthly_fee:,.2f}/mes**.")
    else:
        st.info("Ningún cambio individual a nuestro alcance reduce el riesgo de abandono de este cliente.")

    has_internet = customer_values["InternetService"] != "No"
    simulated_values = dict(customer_values)
    widget_columns = st.columns(4)
    for widget_column, variable in zip(widget_columns, ["Contract", "TechSupport", "OnlineSecurity", "PaymentMethod"]):
        options = ["No", "Yes"] if variable in ["TechSupport", "OnlineSecurity"] and has_internet else category_values[variable]
        simulated_values[variable] = widget_column.selectbox(
            f"{PLAIN_LANGUAGE_LABELS[variable]} (simulado)", options, index=options.index(customer_values[variable]),
            format_func=display_value,
            disabled=variable in ["TechSupport", "OnlineSecurity"] and not has_internet,
            key=f"what_if_{variable}_{hash(tuple(customer_values.items()))}")  # new customer -> widgets reset

    simulated_probability = churn_probabilities_for([simulated_values])[0]
    current_segment, simulated_segment = risk_segment_of(current_probability), risk_segment_of(simulated_probability)
    money_at_risk_change = (simulated_probability - current_probability) * monthly_fee
    result_columns = st.columns(4)
    result_columns[0].metric("Dinero en riesgo hoy", f"${money_at_risk_now:,.2f}/mes",
                             help="Probabilidad de abandono × cuota mensual")
    risk_change_points = round((simulated_probability - current_probability) * 100)
    result_columns[1].metric("Riesgo tras el cambio", f"{simulated_probability:.0%}",  # no badge when nothing changes
                             delta=f"{risk_change_points:+.0f} pts" if risk_change_points != 0 else None, delta_color="inverse")
    result_columns[2].metric("Segmento tras el cambio", simulated_segment,
                             delta=f"antes: {current_segment}" if simulated_segment != current_segment else None, delta_color="off")
    result_columns[3].metric("Dinero en riesgo tras el cambio", f"${simulated_probability * monthly_fee:,.2f}/mes",
                             delta=(f"{'-' if money_at_risk_change < 0 else '+'}${abs(money_at_risk_change):,.2f}/mes"
                                    if round(money_at_risk_change, 2) != 0 else None),
                             delta_color="inverse")
    st.caption("⚠️ Esto muestra cómo cambia la predicción del modelo, no garantiza que el cliente se quede: el modelo "
               "aprendió asociaciones, no causas. La cuota mensual se mantiene fija. Valida cada acción con un grupo de control.")


# ----------------------------------------------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------------------------------------------
html(brand_header("Predicción de abandono y retención explicable para proveedores de internet medianos"))

TAB_LABELS = ["🏠 Inicio", "📊 Análisis de clientes", "🧪 Desempeño del modelo", "🔍 Predicción y explicabilidad",
              "💼 Aplicación de negocio"]
home_tab, eda_tab, performance_tab, prediction_tab, business_tab = st.tabs(TAB_LABELS)

# ----------------------------------------------------------------------------------------------------------------------
# Home: value proposition, three key figures (read from the exported results) and a short guide
# ----------------------------------------------------------------------------------------------------------------------
with home_tab:
    html('<div class="mn-hero"><div class="mn-eyebrow">Retención de clientes para ISP</div>'
         '<div class="mn-headline">Descubre qué clientes se irán, por qué y qué hacer, '
         '<span>antes de que se vayan.</span></div>'
         '<div class="mn-subhead">Menolyte Core analiza cada cliente, estima su riesgo de abandono, explica las razones '
         'en lenguaje simple y le dice a tu equipo a quién llamar primero.</div></div>')

    # Same definitions as the Business Application tab, on the 1,409 test customers the model never saw
    home_test_customers = app_results["test_predictions"]
    home_high_risk = home_test_customers[home_test_customers["churn_probability"].map(risk_segment_of) == HIGH_RISK_NAME]
    home_contact_share = 0.20
    home_contacted = home_test_customers[home_test_customers["share_of_customers_contacted"] <= home_contact_share]
    home_churners_reached = home_contacted["actual_churn"].sum() / home_test_customers["actual_churn"].sum()
    home_recall = metadata["test_metrics"]["recall"]

    figure_columns = st.columns(3, gap="medium")
    figure_columns[0].markdown(figure_card(
        f"{home_recall:.0%}", "de los clientes que se van, el modelo los detecta a tiempo",
        f"Es decir, unos {home_recall * 10:.0f} de cada 10. Medido en {len(home_test_customers):,} clientes "
        "que el modelo nunca vio."), unsafe_allow_html=True)
    figure_columns[1].markdown(figure_card(
        f"{home_churners_reached:.0%}", f"de los que se van, llamando solo al {home_contact_share:.0%} más riesgoso",
        f"{home_churners_reached / home_contact_share:.1f} veces más efectivo que llamar a clientes al azar."),
        unsafe_allow_html=True)
    figure_columns[2].markdown(figure_card(
        f"${home_high_risk['MonthlyCharges'].sum():,.0f}", "de ingreso mensual en clientes de alto riesgo",
        f"{len(home_high_risk):,} clientes con probabilidad de abandono ≥ {risk_segment_edges[2]:.0%}: "
        "los primeros a proteger."), unsafe_allow_html=True)

    html('<div class="mn-section-title">Cómo usar Menolyte Core</div>')
    step_columns = st.columns(4, gap="medium")
    how_to_steps = [
        ("Conoce a tus clientes", TAB_LABELS[1], "Filtra por contrato, servicio o antigüedad y mira quiénes se van más."),
        ("Confía en el modelo", TAB_LABELS[2], "Revisa qué tan bien acierta con clientes que nunca vio."),
        ("Entiende a un cliente", TAB_LABELS[3], "Ingresa sus datos: verás su riesgo, las razones y qué acción lo baja."),
        ("Actúa", TAB_LABELS[4], "Elige cuántos clientes puede llamar tu equipo y descarga la lista priorizada."),
    ]
    for step_number, (step_column, (title, tab_name, text)) in enumerate(zip(step_columns, how_to_steps), start=1):
        step_column.markdown(step_card(step_number, title, tab_name, text), unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------
# A. Exploratory Data Analysis
# ----------------------------------------------------------------------------------------------------------------------
with eda_tab:
    st.subheader("¿Quiénes se van? Conoce tu base de clientes")
    st.caption("Filtra un grupo de clientes y compara su tasa de abandono con la de toda la base.")
    filter_columns = st.columns(4)
    selected_contracts = filter_columns[0].multiselect("Contrato", category_values["Contract"], default=category_values["Contract"],
                                                      format_func=display_value)
    selected_internet = filter_columns[1].multiselect("Servicio de internet", category_values["InternetService"],
                                                      default=category_values["InternetService"], format_func=display_value)
    selected_payment = filter_columns[2].multiselect("Método de pago", category_values["PaymentMethod"],
                                                     default=category_values["PaymentMethod"], format_func=display_value)
    minimum_tenure, maximum_tenure = int(customer_data["tenure"].min()), int(customer_data["tenure"].max())
    selected_tenure = filter_columns[3].slider("Meses como cliente", minimum_tenure, maximum_tenure,
                                               (minimum_tenure, maximum_tenure))

    segment_data = customer_data[customer_data["Contract"].isin(selected_contracts)
                                 & customer_data["InternetService"].isin(selected_internet)
                                 & customer_data["PaymentMethod"].isin(selected_payment)
                                 & customer_data["tenure"].between(*selected_tenure)]

    if segment_data.empty:
        st.warning("Ningún cliente cumple estos filtros.")
    else:
        overall_churn_rate = customer_data["churned"].mean()
        segment_churn_rate = segment_data["churned"].mean()
        kpi_columns = st.columns(4)
        kpi_columns[0].metric("Clientes en el grupo", f"{len(segment_data):,}")
        churn_rate_gap_points = round((segment_churn_rate - overall_churn_rate) * 100, 1)
        kpi_columns[1].metric("Tasa de abandono", f"{segment_churn_rate:.1%}",  # no delta badge when the gap rounds to zero
                              delta=f"{churn_rate_gap_points:+.1f} pts vs. toda la base" if churn_rate_gap_points != 0 else None,
                              delta_color="inverse")
        kpi_columns[2].metric("Cuota mensual promedio", f"${segment_data['MonthlyCharges'].mean():,.2f}")
        kpi_columns[3].metric("Ingreso mensual perdido por abandono",
                              f"${segment_data.loc[segment_data['churned'] == 1, 'MonthlyCharges'].sum():,.0f}")

        stay_leave_names = {"No": "Se queda", "Yes": "Se va"}
        chart_columns = st.columns(2)
        with chart_columns[0]:
            breakdown_variable = st.selectbox("Ver la tasa de abandono por", categorical_features,
                                              index=categorical_features.index("Contract"),
                                              format_func=lambda variable: PLAIN_LANGUAGE_LABELS.get(variable, variable))
            breakdown_name = PLAIN_LANGUAGE_LABELS.get(breakdown_variable, breakdown_variable)
            churn_by_category = (segment_data.groupby(breakdown_variable)["churned"].agg(["mean", "size"])
                                 .reset_index().rename(columns={"mean": "churn_rate", "size": "customers"}))
            churn_by_category[breakdown_variable] = churn_by_category[breakdown_variable].astype(str).map(display_value)
            # Color only: red = above the whole base's churn rate, grey = below (one message: where to look)
            figure = px.bar(churn_by_category, x=breakdown_variable, y="churn_rate", text_auto=".0%",
                            hover_data=["customers"],
                            color=np.where(churn_by_category["churn_rate"] > overall_churn_rate, "above", "below"),
                            color_discrete_map={"above": HIGHLIGHT_COLOR, "below": NEUTRAL_COLOR},
                            labels={"churn_rate": "Tasa de abandono", "customers": "Clientes", breakdown_variable: breakdown_name})
            figure.add_hline(y=overall_churn_rate, line_dash="dash", line_color="#2d2d2d",
                             annotation_text="Promedio de toda la base", annotation_position="top right")
            figure.update_layout(yaxis_tickformat=".0%", yaxis_title="Tasa de abandono", xaxis_title="",
                                 showlegend=False, title=f"Tasa de abandono por {breakdown_name.lower()}")
            st.plotly_chart(figure, width="stretch", theme=None)
        with chart_columns[1]:
            figure = px.histogram(segment_data, x="tenure", color="Churn", barmode="overlay", histnorm="percent", nbins=36,
                                  color_discrete_map={"Yes": HIGHLIGHT_COLOR, "No": NEUTRAL_COLOR},
                                  labels={"tenure": "Meses como cliente", "Churn": "¿Se fue?"},
                                  title="Antigüedad: quién se queda y quién se va")
            figure.for_each_trace(lambda trace: trace.update(name=stay_leave_names[trace.name]))
            figure.update_layout(xaxis_title="Meses como cliente", yaxis_title="% del grupo", legend_title="")
            st.plotly_chart(figure, width="stretch", theme=None)

        chart_columns = st.columns(2)
        with chart_columns[0]:
            # Contract × tenure band: where churn concentrates (the two variables that weigh most for the model)
            tenure_band_labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
            banded = segment_data.assign(tenure_band=pd.cut(segment_data["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                                            labels=tenure_band_labels))
            contract_order = [contract for contract in category_values["Contract"] if contract in set(banded["Contract"])]
            churn_grid = banded.pivot_table(index="Contract", columns="tenure_band", values="churned", aggfunc="mean",
                                            observed=False).reindex(index=contract_order, columns=tenure_band_labels)
            size_grid = banded.pivot_table(index="Contract", columns="tenure_band", values="churned", aggfunc="size",
                                           observed=False).reindex(index=contract_order, columns=tenure_band_labels)
            figure = go.Figure(go.Heatmap(
                z=churn_grid.to_numpy(), x=tenure_band_labels, y=[display_value(contract) for contract in contract_order],
                customdata=size_grid.fillna(0).to_numpy(), zmin=0, zmax=max(float(np.nanmax(churn_grid.to_numpy())), 0.01),
                colorscale=[[0, "#ffffff"], [0.35, "#fbe3e3"], [0.7, "#f19c9c"], [1, "#e05a5b"]],
                texttemplate="%{z:.0%}", textfont={"size": 15, "color": "#2d2d2d"}, xgap=4, ygap=4, showscale=False,
                hovertemplate="%{y}, %{x} meses<br>Tasa de abandono: %{z:.1%}<br>Clientes: %{customdata:,}<extra></extra>"))
            figure.update_layout(title="Dónde se concentra el abandono: contrato × antigüedad",
                                 xaxis_title="Meses como cliente", yaxis={"autorange": "reversed", "showgrid": False},
                                 xaxis={"showgrid": False})
            st.plotly_chart(figure, width="stretch", theme=None)
        with chart_columns[1]:
            # Money, not rates: monthly revenue of the customers who left, by contract and internet type
            lost_revenue = (segment_data[segment_data["churned"] == 1]
                            .groupby(["Contract", "InternetService"])["MonthlyCharges"].sum().reset_index())
            if lost_revenue.empty:
                st.info("Ningún cliente de este grupo se fue: no hay ingreso perdido.")
            else:
                lost_revenue["Contrato"] = lost_revenue["Contract"].map(display_value)
                lost_revenue["Internet"] = lost_revenue["InternetService"].map(display_value)
                figure = px.bar(lost_revenue, x="MonthlyCharges", y="Contrato", color="Internet", orientation="h",
                                barmode="stack", text_auto="$,.0f",
                                category_orders={"Contrato": [display_value(contract) for contract in category_values["Contract"]],
                                                 "Internet": ["Fibra óptica", "DSL", "No"]},
                                color_discrete_map={"Fibra óptica": HIGHLIGHT_COLOR, "DSL": "#6b7280", "No": NEUTRAL_COLOR},
                                labels={"MonthlyCharges": "Ingreso mensual perdido"},
                                title="¿Dónde se pierde el dinero? Ingreso mensual perdido")
                # Labels only where they fit (no rotated/clipped text) + the total at the end of each bar
                figure.update_traces(textposition="inside", insidetextanchor="middle", textangle=0)
                lost_by_contract = lost_revenue.groupby("Contrato")["MonthlyCharges"].sum()
                for contract_name, total in lost_by_contract.items():
                    figure.add_annotation(x=total, y=contract_name, text=f"<b>${total:,.0f}</b>", showarrow=False,
                                          xanchor="left", xshift=6, font={"size": 13, "color": "#2d2d2d"})
                figure.update_layout(xaxis_title="Ingreso mensual perdido (USD)", yaxis_title="", legend_title="Internet",
                                     xaxis_tickprefix="$", uniformtext_minsize=11, uniformtext_mode="hide",
                                     xaxis_range=[0, lost_by_contract.max() * 1.22])
                st.plotly_chart(figure, width="stretch", theme=None)

        chart_columns = st.columns(2)
        with chart_columns[0]:
            # Churn with vs without each add-on (customers with internet only): basis of the "free trial" actions
            internet_customers = segment_data[segment_data["InternetService"] != "No"]
            add_on_rates = [(PLAIN_LANGUAGE_LABELS[add_on],
                             internet_customers.loc[internet_customers[add_on] == "Yes", "churned"].mean(),
                             internet_customers.loc[internet_customers[add_on] == "No", "churned"].mean())
                            for add_on in ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection"]]
            add_on_rates = [row for row in add_on_rates if not (np.isnan(row[1]) or np.isnan(row[2]))]
            if not add_on_rates:
                st.info("Este grupo no tiene clientes con internet para comparar servicios adicionales.")
            else:
                add_on_names, with_rates, without_rates = zip(*add_on_rates)
                figure = go.Figure()
                for name, with_rate, without_rate in add_on_rates:
                    figure.add_trace(go.Scatter(x=[with_rate, without_rate], y=[name, name], mode="lines",
                                                line={"color": "#dfe3e8", "width": 6}, showlegend=False, hoverinfo="skip"))
                figure.add_trace(go.Scatter(x=with_rates, y=add_on_names, mode="markers+text", name="Con el servicio",
                                            marker={"color": "#6b7280", "size": 16}, text=[f"{rate:.0%}" for rate in with_rates],
                                            textposition="bottom center", hovertemplate="Con el servicio: %{x:.1%}<extra></extra>"))
                figure.add_trace(go.Scatter(x=without_rates, y=add_on_names, mode="markers+text", name="Sin el servicio",
                                            marker={"color": HIGHLIGHT_COLOR, "size": 16},
                                            text=[f"{rate:.0%}" for rate in without_rates], textposition="bottom center",
                                            hovertemplate="Sin el servicio: %{x:.1%}<extra></extra>"))
                figure.update_layout(title="Abandono con y sin cada servicio adicional (clientes con internet)",
                                     xaxis_title="Tasa de abandono", xaxis_tickformat=".0%", yaxis_title="",
                                     xaxis_rangemode="tozero")
                st.plotly_chart(figure, width="stretch", theme=None)
        with chart_columns[1]:
            figure = px.box(segment_data, x="Churn", y="MonthlyCharges", color="Churn",
                            color_discrete_map={"Yes": HIGHLIGHT_COLOR, "No": NEUTRAL_COLOR},
                            labels={"MonthlyCharges": "Cuota mensual", "Churn": "¿Se fue?"},
                            title="Cuota mensual de quienes se quedan vs. quienes se van")
            figure.update_xaxes(tickvals=list(stay_leave_names), ticktext=list(stay_leave_names.values()), title="")
            figure.update_layout(yaxis_title="Cuota mensual (USD)", showlegend=False)
            st.plotly_chart(figure, width="stretch", theme=None)

# ----------------------------------------------------------------------------------------------------------------------
# B. Model Performance
# ----------------------------------------------------------------------------------------------------------------------
with performance_tab:
    st.subheader("¿Qué tan bueno es el modelo?")
    hyperparameters = metadata["hyperparameters"]
    st.caption("Medido en 1,409 clientes que el modelo nunca vio. Métrica principal: **F2**, que da el doble de peso a "
               "*detectar* a quien se va (recall) que a no equivocarse al marcarlo (precisión), porque no detectar a un "
               "cliente que se va es el error más caro.")

    metric_display = {
        "f2": ("F2 · métrica principal", "Combina recall y precisión, dando el doble de peso al recall."),
        "recall": ("Recall · detectados", "De los clientes que se fueron, qué parte detectó el modelo."),
        "precision": ("Precisión · aciertos", "De los clientes marcados como 'se va', qué parte se fue de verdad."),
        "f1": ("F1 · balance", "Combina recall y precisión con el mismo peso."),
        "roc_auc": ("ROC-AUC · orden", "Qué tan bien ordena a los clientes por riesgo (1 = perfecto, 0.5 = al azar)."),
    }
    metric_columns = st.columns(5)
    for column, metric_name in zip(metric_columns, ["f2", "recall", "precision", "f1", "roc_auc"]):
        test_value = metadata["test_metrics"][metric_name]
        cross_validation_value = metadata["cross_validation_metrics"][metric_name]
        metric_label, metric_help = metric_display[metric_name]
        column.metric(metric_label, f"{test_value:.3f}",
                      delta=f"{test_value - cross_validation_value:+.3f} vs. validación cruzada", help=metric_help,
                      delta_color="off")  # grey: a small test-vs-CV gap is expected, not an alarm

    chart_columns = st.columns(2)
    with chart_columns[0]:
        confusion_matrix = app_results["test_confusion_matrix"].to_numpy()  # rows: actual (stays, churns); cols: predicted
        # Color by meaning, not by count: correct = grey, false alarm = light red, missed churner (costliest) = red
        cell_kinds = [["Acierto", "Falsa alarma"], ["No detectado", "Detectado"]]
        cell_color_codes = [[0, 1], [2, 0]]
        predicted_labels, actual_labels = ["Predijo: se queda", "Predijo: se va"], ["Se quedó", "Se fue"]
        figure = go.Figure(go.Heatmap(z=cell_color_codes, x=predicted_labels, y=actual_labels, zmin=0, zmax=2,
                                      colorscale=[[0, "#eef0f3"], [0.5, "#f9d4d4"], [1, HIGHLIGHT_COLOR]],
                                      showscale=False, xgap=6, ygap=6, hoverinfo="skip"))
        for row in range(2):
            for column in range(2):
                figure.add_annotation(x=predicted_labels[column], y=actual_labels[row], showarrow=False,
                                      text=f"<b>{confusion_matrix[row, column]:,}</b><br>{cell_kinds[row][column]}",
                                      font={"size": 18, "color": "#ffffff" if cell_color_codes[row][column] == 2 else "#2d2d2d"})
        figure.update_layout(title=f"{confusion_matrix[1, 0]:,} de {confusion_matrix[1].sum():,} clientes que se iban "
                                   "pasaron sin detectar",
                             yaxis={"autorange": "reversed", "title": "En realidad", "showgrid": False},
                             xaxis={"side": "bottom", "showgrid": False})
        st.plotly_chart(figure, width="stretch", theme=None)
    with chart_columns[1]:
        model_comparison = app_results["model_comparison_cv"]
        metric_means = (model_comparison[[f"{metric}_mean" for metric in ["f2", "recall", "precision", "f1", "roc_auc"]]]
                        .rename(columns=lambda column: column.replace("_mean", "").upper())
                        .reset_index(names="model").melt(id_vars="model", var_name="metric", value_name="score"))
        figure = px.bar(metric_means, x="metric", y="score", color="model", barmode="group",
                        color_discrete_map={"SVM (RBF)": HIGHLIGHT_COLOR, "Random Forest": NEUTRAL_COLOR,
                                            "Logistic Regression": "#6b7280"},
                        labels={"metric": "métrica", "score": "puntaje", "model": "modelo"},
                        title="El SVM elegido frente a otros modelos")
        figure.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="Puntaje (validación cruzada)", legend_title="")
        st.plotly_chart(figure, width="stretch", theme=None)

    st.markdown("#### ¿Podemos creerle al porcentaje?")
    test_scores = app_results["test_predictions"]
    chart_columns = st.columns(2)
    with chart_columns[0]:
        # Calibration: mean predicted probability vs actual churn rate, by probability band (test customers)
        calibration = (test_scores.assign(band=pd.cut(test_scores["churn_probability"],
                                                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True))
                       .groupby("band", observed=True)
                       .agg(predicted=("churn_probability", "mean"), actual=("actual_churn", "mean"),
                            customers=("actual_churn", "size")).reset_index())
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": NEUTRAL_COLOR, "dash": "dash"},
                                    name="Calibración perfecta", hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=calibration["predicted"], y=calibration["actual"], mode="lines+markers",
                                    line={"color": HIGHLIGHT_COLOR, "width": 3},
                                    marker={"color": HIGHLIGHT_COLOR, "size": np.sqrt(calibration["customers"]) * 1.6,
                                            "line": {"color": "#ffffff", "width": 1.5}},
                                    customdata=calibration["customers"], name="Menolyte Core",
                                    hovertemplate="El modelo dijo %{x:.0%}<br>Se fueron %{y:.0%}<br>Clientes: %{customdata:,}"
                                                  "<extra></extra>"))
        figure.update_layout(title="Cuando el modelo dice 70%, se van unos 7 de cada 10",
                             xaxis_title="Probabilidad que dijo el modelo", yaxis_title="Clientes que se fueron de verdad",
                             xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(figure, width="stretch", theme=None)
    with chart_columns[1]:
        # Separation: distribution of predicted probabilities for customers who stayed vs left
        figure = px.histogram(test_scores.assign(outcome=test_scores["actual_churn"].map({0: "Se quedó", 1: "Se fue"})),
                              x="churn_probability", color="outcome", barmode="overlay", histnorm="percent", nbins=20,
                              color_discrete_map={"Se quedó": NEUTRAL_COLOR, "Se fue": HIGHLIGHT_COLOR},
                              category_orders={"outcome": ["Se quedó", "Se fue"]},
                              labels={"churn_probability": "probabilidad", "outcome": "resultado"},
                              title="El modelo separa a quienes se quedan de quienes se van")
        for edge in risk_segment_edges[1:-1]:
            figure.add_vline(x=edge, line_dash="dot", line_color="#2d2d2d")
        figure.update_layout(xaxis_title="Probabilidad de abandono predicha", yaxis_title="% del grupo",
                             xaxis_tickformat=".0%", legend_title="")
        st.plotly_chart(figure, width="stretch", theme=None)
    st.caption(f"Ambos gráficos usan los {len(test_scores):,} clientes de prueba. Izquierda: cada punto agrupa clientes con "
               "probabilidad parecida (tamaño = cantidad); cerca de la diagonal = el porcentaje es confiable. Derecha: "
               f"líneas punteadas = límites de los segmentos de riesgo ({risk_segment_edges[1]:.0%} y {risk_segment_edges[2]:.0%}).")

    st.markdown("#### ¿Cuántos datos hay que pedir por cliente?")
    # Slot reserved here, filled at the END of the script: the slow first computation no longer blocks the other tabs
    form_curve_slot = st.container()

    with st.expander("🔧 Detalles técnicos: optimización e importancia de variables"):
        st.caption(f"Modelo final: SVM (kernel RBF) ajustado con Grid Search. C = {hyperparameters['C']:.3f}, "
                   f"gamma = {hyperparameters['gamma']:.4f}, class_weight = balanced.")

        st.markdown("#### Optimización de hiperparámetros (mismo modelo, particiones y presupuesto de 36 configuraciones)")
        strategies = app_results["optimization_strategies"].reset_index(names="strategy")
        st.dataframe(strategies[["strategy", "configurations_evaluated", "optimization_seconds", "best_C", "best_gamma",
                                 "cv_f2", "cv_recall", "cv_precision", "f2_gain_vs_default_svm"]].round(4)
                     .rename(columns={"strategy": "estrategia", "configurations_evaluated": "configuraciones",
                                      "optimization_seconds": "segundos", "best_C": "mejor C", "best_gamma": "mejor gamma",
                                      "cv_f2": "F2 (VC)", "cv_recall": "recall (VC)", "cv_precision": "precisión (VC)",
                                      "f2_gain_vs_default_svm": "ganancia F2 vs. SVM base"}),
                     hide_index=True, width="stretch")
        chart_columns = st.columns(2)
        with chart_columns[0]:
            figure = px.bar(strategies, x="strategy", y="cv_f2", text_auto=".4f", title="Mejor F2 en validación cruzada",
                            color="strategy", color_discrete_map={"Grid Search": HIGHLIGHT_COLOR},
                            labels={"strategy": "estrategia", "cv_f2": "F2"},
                            color_discrete_sequence=[NEUTRAL_COLOR])
            figure.update_layout(yaxis_range=[strategies["cv_f2"].min() - 0.02, strategies["cv_f2"].max() + 0.01],
                                 xaxis_title="", yaxis_title="F2", showlegend=False)
            st.plotly_chart(figure, width="stretch", theme=None)
        with chart_columns[1]:
            figure = px.bar(strategies, x="strategy", y="optimization_seconds", text_auto=".1f",
                            title="Tiempo de optimización (segundos)", color_discrete_sequence=[NEUTRAL_COLOR],
                            labels={"strategy": "estrategia", "optimization_seconds": "segundos"})
            figure.update_layout(xaxis_title="", yaxis_title="Segundos")
            st.plotly_chart(figure, width="stretch", theme=None)

        st.markdown("#### Optimización multiobjetivo (NSGA-II): F2 vs. tiempo de inferencia")
        nsga2_trials = app_results["nsga2_trials"]
        pareto_front = nsga2_trials[nsga2_trials["is_pareto_optimal"]].sort_values("ms_per_1000_customers")
        grid_best = app_results["grid_best_speed"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=nsga2_trials.loc[~nsga2_trials["is_pareto_optimal"], "ms_per_1000_customers"],
                                    y=nsga2_trials.loc[~nsga2_trials["is_pareto_optimal"], "cv_f2"], mode="markers",
                                    marker={"color": "#dfe3e8", "size": 9}, name="Configuraciones dominadas"))
        figure.add_trace(go.Scatter(x=pareto_front["ms_per_1000_customers"], y=pareto_front["cv_f2"], mode="lines+markers",
                                    marker={"color": HIGHLIGHT_COLOR, "size": 11}, line={"color": HIGHLIGHT_COLOR},
                                    name="Frente de Pareto", customdata=pareto_front[["C", "gamma"]],
                                    hovertemplate="F2 %{y:.4f}<br>%{x:.0f} ms<br>C %{customdata[0]:.3f}, gamma %{customdata[1]:.4f}"))
        figure.add_trace(go.Scatter(x=[grid_best["ms_per_1000_customers"]], y=[grid_best["cv_f2"]], mode="markers",
                                    marker={"symbol": "star", "size": 22, "color": "#2d2d2d", "line": {"color": "#ffffff", "width": 1.5}},
                                    name="Modelo elegido (ganador de Grid Search)"))
        figure.update_layout(xaxis_title="Tiempo de inferencia (ms por 1,000 clientes): menos es mejor",
                             yaxis_title="F2 en validación cruzada: más es mejor", height=450)
        st.plotly_chart(figure, width="stretch", theme=None)
        st.caption("El modelo elegido no está dominado: está en el extremo de mayor F2 del frente. Las soluciones más "
                   "rápidas del 'codo' ahorran unas centésimas de milisegundo por cliente, irrelevante a esta escala.")

        st.markdown("#### Qué variables pesan más: SHAP (SVM elegido) vs. importancia de Random Forest")
        importance = app_results["importance_comparison"][["SHAP (SVM)", "Random Forest MDI"]].iloc[::-1].reset_index(names="variable")
        importance["variable"] = importance["variable"].map(lambda variable: PLAIN_LANGUAGE_LABELS.get(variable, variable))
        figure = px.bar(importance.melt(id_vars="variable", var_name="method", value_name="normalized importance"),
                        x="normalized importance", y="variable", color="method", barmode="group", orientation="h", height=600,
                        color_discrete_map={"SHAP (SVM)": HIGHLIGHT_COLOR, "Random Forest MDI": NEUTRAL_COLOR},
                        labels={"normalized importance": "importancia normalizada", "method": "método"})
        figure.update_layout(xaxis_title="Importancia normalizada", yaxis_title="", legend_title="")
        st.plotly_chart(figure, width="stretch", theme=None)

        st.markdown("#### Grid Search: F2 en validación cruzada para cada combinación de C y gamma")
        f2_grid = app_results["grid_search_f2_matrix"]  # rows: gamma, columns: C (exported by the notebook)
        figure = go.Figure(go.Heatmap(
            z=f2_grid.to_numpy(), x=[str(value) for value in f2_grid.columns], y=[str(value) for value in f2_grid.index],
            # Color range starts at 0.68: failed configs (F2 = 0) would otherwise flatten all the good cells into one red
            zmin=0.68, zmax=float(np.nanmax(f2_grid.to_numpy())),
            colorscale=[[0, "#ffffff"], [0.4, "#fbe3e3"], [0.75, "#f19c9c"], [1, "#e05a5b"]], texttemplate="%{z:.4f}",
            textfont={"size": 13, "color": "#2d2d2d"}, xgap=3, ygap=3, showscale=False,
            hovertemplate="C = %{x}<br>gamma = %{y}<br>F2 = %{z:.4f}<extra></extra>"))
        best_gamma, best_c = np.unravel_index(np.nanargmax(f2_grid.to_numpy()), f2_grid.shape)
        figure.add_shape(type="rect", x0=best_c - 0.5, x1=best_c + 0.5, y0=best_gamma - 0.5, y1=best_gamma + 0.5,
                         line={"color": "#2d2d2d", "width": 3})
        figure.update_layout(title="La combinación elegida (recuadro) tiene el mejor F2 de la cuadrícula", height=430,
                             xaxis={"title": "C (regularización)", "type": "category", "showgrid": False},
                             yaxis={"title": "gamma (alcance del kernel)", "type": "category", "showgrid": False})
        st.plotly_chart(figure, width="stretch", theme=None)

        st.markdown("#### Impacto de negocio de cada modelo (validación cruzada, hiperparámetros por defecto)")
        business_impact = app_results["business_impact_cv"].rename(columns={
            "churners_detected (TP)": "clientes que se van detectados", "churners_missed (FN)": "no detectados",
            "unnecessary_contacts (FP)": "contactos innecesarios", "monthly_revenue_reachable_TP ($)": "ingreso mensual alcanzable",
            "monthly_revenue_lost_FN ($)": "ingreso mensual no detectado", "total_contacts": "contactos totales",
            "contacts_per_churner_found": "contactos por cliente encontrado",
            "hypothetical_cost_of_wasted_contacts ($)": "costo hipotético de contactos innecesarios"})
        money_rows = ["ingreso mensual alcanzable", "ingreso mensual no detectado", "costo hipotético de contactos innecesarios"]
        impact_display = business_impact.T.apply(  # models as columns: fits the page and compares side by side
            lambda row: row.map((lambda value: f"${value:,.0f}") if row.name in money_rows
                                else (lambda value: f"{value:.2f}") if row.name == "contactos por cliente encontrado"
                                else (lambda value: f"{value:,.0f}")), axis=1)
        st.dataframe(impact_display, width="stretch")

# ----------------------------------------------------------------------------------------------------------------------
# C. Prediction & Explainability
# ----------------------------------------------------------------------------------------------------------------------
with prediction_tab:
    st.subheader("Evalúa a un cliente y entiende por qué")

    test_predictions = app_results["test_predictions"]
    example_customers = {
        "Cliente nuevo típico (ingreso manual)": None,
        "Ejemplo: cliente de prueba con mayor riesgo": test_predictions.iloc[0]["customer_index"],
        "Ejemplo: cliente de prueba con menor riesgo": test_predictions.iloc[-1]["customer_index"],
    }
    example_choice = st.selectbox("Partir de", list(example_customers))
    example_index = example_customers[example_choice]
    if example_index is None:
        default_values = {"Contract": "Month-to-month", "tenure": 3, "MonthlyCharges": 85.0, "PaymentMethod": "Electronic check",
                          "InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No"}
    else:
        default_values = customer_data.loc[int(example_index), FORM_FIELDS].to_dict()

    with st.form("customer_form"):
        input_values = {}
        form_columns = st.columns(2)
        for form_column, (group_title, group_variables) in zip(form_columns, FORM_FIELD_GROUPS.items()):
            form_column.markdown(f"**{group_title}**")
            for variable in group_variables:
                widget_key = f"{example_choice}_{variable}"
                label = PLAIN_LANGUAGE_LABELS[variable]
                if variable == "tenure":
                    input_values[variable] = form_column.slider(label + " (meses)", 0, 72, int(default_values[variable]), key=widget_key)
                elif variable == "MonthlyCharges":
                    input_values[variable] = form_column.number_input(label + " (USD)", 0.0, 10000.0,
                                                                      float(default_values[variable]), key=widget_key)
                else:
                    options = ["No", "Yes"] if variable in INTERNET_ADD_ONS else category_values[variable]
                    default_value = "No" if default_values[variable] == "No internet service" else default_values[variable]
                    input_values[variable] = form_column.selectbox(label, options, index=options.index(default_value),
                                                                   format_func=display_value, key=widget_key)
        st.caption(f"Solo los {len(FORM_FIELDS)} datos que más pesan para el modelo (ver Desempeño del modelo). El resto toma "
                   "el valor del cliente típico; total pagado = meses × cuota mensual. Seguridad en línea y soporte "
                   "técnico se ignoran para clientes sin internet.")
        submitted = st.form_submit_button("Predecir riesgo de abandono", type="primary")

    # Keep the scored customer across reruns (what-if widgets, other tabs), one per "Start from" option
    if submitted:
        st.session_state[f"scored_customer_{example_choice}"] = complete_customers(pd.DataFrame([input_values]),
                                                                                   FORM_FIELDS).to_dict("records")[0]
    scored_customer = st.session_state.get(f"scored_customer_{example_choice}")
    if scored_customer is not None:
        input_values = scored_customer
        customer_row = pd.DataFrame([input_values])[feature_columns]
        churn_probability = final_pipeline.predict_proba(customer_row)[0, 1]
        predicted_churn = final_pipeline.predict(customer_row)[0]
        risk_segment = risk_segment_of(churn_probability)

        # Result card: risk scale + segment badge + decision in plain words (same values as before, new presentation)
        risk_level = RISK_SEGMENT_NAMES.index(risk_segment)
        with st.container(border=True, key=f"prediction_card_{risk_level}"):
            scale_column, summary_column = st.columns([1.15, 1], gap="large")
            scale_column.markdown(risk_scale(churn_probability, risk_segment_edges,
                                             [name.replace("Riesgo ", "").capitalize() for name in RISK_SEGMENT_NAMES],
                                             risk_level), unsafe_allow_html=True)
            summary_column.markdown(
                f'<div class="mn-result-label">Segmento de riesgo</div>{risk_badge(risk_segment, risk_level)}'
                f'<div class="mn-result-label">Decisión del modelo</div>'
                f'<div class="mn-result-decision">{"Probablemente se va" if predicted_churn == 1 else "Probablemente se queda"}</div>'
                f'<div class="mn-result-plain">{plain_probability_sentence(churn_probability)}</div>', unsafe_allow_html=True)

        with st.spinner("Explicando la predicción con SHAP y LIME..."):
            shap_by_variable, lime_conditions = explain_customer(input_values)

        top_drivers = shap_by_variable.reindex(shap_by_variable.abs().sort_values(ascending=False).index).head(8)

        st.markdown("#### ¿Por qué? Los principales factores detrás de esta predicción")
        explanation_columns = st.columns(2)
        with explanation_columns[0]:
            shap_chart_data = pd.DataFrame({
                "factor": [f"{PLAIN_LANGUAGE_LABELS[variable]}: {describe_value(variable, input_values[variable])}"
                           for variable in top_drivers.index],
                "risk points": top_drivers.values * 100,
            }).iloc[::-1]
            figure = px.bar(shap_chart_data, x="risk points", y="factor", orientation="h", text_auto=".1f",
                            color=np.where(shap_chart_data["risk points"] > 0, "Sube el riesgo", "Baja el riesgo"),
                            color_discrete_map={"Sube el riesgo": HIGHLIGHT_COLOR, "Baja el riesgo": RETENTION_COLOR},
                            labels={"risk points": "puntos de riesgo", "factor": "factor"},
                            title=f"SHAP: puntos de riesgo sumados al cliente promedio ({metadata['shap_base_value']:.0%})")
            figure.update_layout(xaxis_title="Puntos de riesgo de abandono", yaxis_title="", legend_title="")
            st.plotly_chart(figure, width="stretch", theme=None)
        with explanation_columns[1]:
            lime_weights = pd.DataFrame(lime_conditions, columns=["condition", "weight"]).iloc[::-1]
            lime_weights["condition"] = lime_weights["condition"].map(describe_lime_condition)  # display only
            figure = px.bar(lime_weights, x="weight", y="condition", orientation="h", text_auto=".2f",
                            color=np.where(lime_weights["weight"] > 0, "Sube el riesgo", "Baja el riesgo"),
                            color_discrete_map={"Sube el riesgo": HIGHLIGHT_COLOR, "Baja el riesgo": RETENTION_COLOR},
                            labels={"weight": "efecto", "condition": "condición"},
                            title="LIME: aproximación lineal local alrededor de este cliente")
            figure.update_layout(xaxis_title="Efecto en la probabilidad de abandono", yaxis_title="", legend_title="")
            st.plotly_chart(figure, width="stretch", theme=None)

        risk_raising_drivers = [variable for variable in top_drivers.index if top_drivers[variable] > 0]
        summary_factors = ", ".join(f"{PLAIN_LANGUAGE_LABELS[variable].lower()} ({describe_value(variable, input_values[variable])})"
                                    for variable in risk_raising_drivers[:3])
        st.markdown("#### En palabras simples")
        if risk_raising_drivers:
            st.info(f"Este cliente tiene una probabilidad de **{churn_probability:.0%}** de irse. "
                    f"Los factores que más suben el riesgo son: {summary_factors}.")
        else:
            st.success(f"Este cliente tiene una probabilidad de **{churn_probability:.0%}** de irse; ningún factor sube el riesgo.")

        actions = [action for action in (suggested_action(variable, input_values[variable]) for variable in risk_raising_drivers) if action]
        if actions and churn_probability >= risk_segment_edges[1]:
            st.markdown("#### Acciones de retención sugeridas")
            # Same actions, same order (drivers ranked by SHAP); top 3 as cards with their reason, the rest folded
            action_reasons = {}
            for variable in risk_raising_drivers:
                action = suggested_action(variable, input_values[variable])
                if action and action not in action_reasons:
                    action_reasons[action] = (f"Porque: {PLAIN_LANGUAGE_LABELS[variable].lower()} "
                                              f"({describe_value(variable, input_values[variable])})")
            prioritized_actions = list(action_reasons.items())
            action_columns = st.columns(3, gap="medium")
            for number, (action_column, (action, reason)) in enumerate(zip(action_columns, prioritized_actions[:3]), start=1):
                action_column.markdown(action_card(number, action, reason), unsafe_allow_html=True)
            if len(prioritized_actions) > 3:
                with st.expander(f"Ver {len(prioritized_actions) - 3} acciones más"):
                    for number, (action, reason) in enumerate(prioritized_actions[3:], start=4):
                        st.markdown(f"**{number}.** {action}  \n<span style='color:#6b7280'>{reason}</span>",
                                    unsafe_allow_html=True)
        st.caption("Las explicaciones muestran lo que el modelo asocia con el abandono, no causas comprobadas. "
                   "Las acciones sugeridas son hipótesis a validar con una prueba controlada (contactados vs. grupo de control).")

        show_what_if_panel(input_values, churn_probability)

# ----------------------------------------------------------------------------------------------------------------------
# D. Business Application
# ----------------------------------------------------------------------------------------------------------------------
with business_tab:
    st.subheader("Menolyte Core para gerentes de retención")
    st.markdown("Cada mes, Menolyte Core evalúa toda la base de clientes, los ordena por riesgo de abandono, explica "
                "cada caso y estima el impacto económico de una campaña de retención. "
                "Abajo: resultados en **1,409 clientes de prueba** que el modelo nunca vio durante el entrenamiento.")

    ranked_customers = app_results["test_predictions"].copy()
    ranked_customers["risk_segment"] = ranked_customers["churn_probability"].map(risk_segment_of)
    high_risk_customers = ranked_customers[ranked_customers["risk_segment"] == HIGH_RISK_NAME]

    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Clientes evaluados", f"{len(ranked_customers):,}")
    kpi_columns[1].metric("Clientes en riesgo alto", f"{len(high_risk_customers):,}",
                          help=f"Probabilidad de abandono ≥ {risk_segment_edges[2]:.0%}")
    kpi_columns[2].metric("Ingreso mensual en riesgo alto", f"${high_risk_customers['MonthlyCharges'].sum():,.0f}")
    kpi_columns[3].metric("Abandono real en riesgo alto", f"{high_risk_customers['actual_churn'].mean():.0%}",
                          help="Parte de los clientes en riesgo alto que efectivamente se fueron (clientes de prueba)")

    st.markdown("#### ¿A cuántos clientes puede contactar tu equipo?")
    contact_share = st.slider("Porcentaje de clientes contactados (los más riesgosos primero)", 5, 100, 20, step=5,
                              format="%d%%") / 100
    contacted_customers = ranked_customers[ranked_customers["share_of_customers_contacted"] <= contact_share]
    churners_reached = contacted_customers["actual_churn"].sum() / ranked_customers["actual_churn"].sum()
    reach_columns = st.columns(3)
    reach_columns[0].metric("Clientes a contactar", f"{len(contacted_customers):,}")
    reach_columns[1].metric("Clientes que se van alcanzados", f"{churners_reached:.0%}",
                            delta=f"{(churners_reached - contact_share) * 100:+.0f} pts vs. elegir al azar")
    reach_columns[2].metric("Ingreso mensual de los que se van alcanzados",
                            f"${contacted_customers.loc[contacted_customers['actual_churn'] == 1, 'MonthlyCharges'].sum():,.0f}")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=ranked_customers["share_of_customers_contacted"], y=ranked_customers["share_of_churners_reached"],
                                    mode="lines", line={"color": HIGHLIGHT_COLOR, "width": 3}, name="Orden de Menolyte Core"))
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": NEUTRAL_COLOR, "dash": "dash"},
                                    name="Elegir al azar"))
        figure.add_vline(x=contact_share, line_dash="dot")
        figure.update_layout(title="Clientes que se van alcanzados vs. clientes contactados", xaxis_tickformat=".0%",
                             yaxis_tickformat=".0%", xaxis_title="Clientes contactados", yaxis_title="De los que se van, alcanzados")
        st.plotly_chart(figure, width="stretch", theme=None)
    with chart_columns[1]:
        segment_summary = (ranked_customers.groupby("risk_segment")
                           .agg(customers=("actual_churn", "size"), actual_churn_rate=("actual_churn", "mean"),
                                monthly_revenue=("MonthlyCharges", "sum"))
                           .reindex(RISK_SEGMENT_NAMES).reset_index())
        figure = px.bar(segment_summary, x="risk_segment", y="actual_churn_rate", text_auto=".0%", color="risk_segment",
                        color_discrete_map=dict(zip(RISK_SEGMENT_NAMES, [NEUTRAL_COLOR, "#f19c9c", HIGHLIGHT_COLOR])),
                        hover_data=["customers", "monthly_revenue"], title="Abandono real por segmento de riesgo",
                        labels={"risk_segment": "segmento", "actual_churn_rate": "abandono real", "customers": "clientes",
                                "monthly_revenue": "ingreso mensual"})
        figure.update_layout(yaxis_tickformat=".0%", showlegend=False, xaxis_title="", yaxis_title="Se fueron")
        st.plotly_chart(figure, width="stretch", theme=None)

    st.markdown("#### ¿Cuánto gana la campaña?")
    # Exported by the notebook (B4): cumulative net benefit when contacting the riskiest customers first
    assumptions = metadata["hypothetical_assumptions"]
    best_campaign = ranked_customers.loc[ranked_customers["cumulative_net_benefit"].idxmax()]
    chosen_campaign_benefit = contacted_customers["cumulative_net_benefit"].iloc[-1]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=ranked_customers["share_of_customers_contacted"], y=ranked_customers["cumulative_net_benefit"],
                                mode="lines", line={"color": HIGHLIGHT_COLOR, "width": 3}, name="Beneficio neto",
                                hovertemplate="Contactando al %{x:.0%}: $%{y:,.0f} al año<extra></extra>"))
    figure.add_trace(go.Scatter(x=[best_campaign["share_of_customers_contacted"]], y=[best_campaign["cumulative_net_benefit"]],
                                mode="markers", marker={"symbol": "star", "size": 18, "color": "#2d2d2d"},
                                name=f"Máximo: ${best_campaign['cumulative_net_benefit']:,.0f} "
                                     f"(al {best_campaign['share_of_customers_contacted']:.0%})",
                                hovertemplate="Máximo: $%{y:,.0f} al %{x:.0%}<extra></extra>"))
    figure.add_trace(go.Scatter(x=[contact_share], y=[chosen_campaign_benefit], mode="markers",
                                marker={"size": 14, "color": HIGHLIGHT_COLOR, "line": {"color": "#ffffff", "width": 2}},
                                name=f"Tu selección: ${chosen_campaign_benefit:,.0f} (al {contact_share:.0%})",
                                hovertemplate="Tu selección: $%{y:,.0f}<extra></extra>"))
    figure.add_vline(x=contact_share, line_dash="dot")
    figure.update_layout(title=f"Contactando al {contact_share:.0%} más riesgoso se ganan ~${chosen_campaign_benefit:,.0f} "
                               f"al año ({chosen_campaign_benefit / best_campaign['cumulative_net_benefit']:.0%} del máximo)",
                         xaxis_title="Clientes contactados (los más riesgosos primero)", yaxis_title="Beneficio neto anual (USD)",
                         xaxis_tickformat=".0%", yaxis_tickprefix="$", height=420)
    st.plotly_chart(figure, width="stretch", theme=None)
    st.caption(f"⚠️ Estimación ilustrativa con supuestos hipotéticos del notebook: costo de ${assumptions['cost_per_contact']} "
               f"por contacto; el {assumptions['retention_success_rate']:.0%} de los clientes contactados que iban a irse se "
               f"quedan; se recuperan {assumptions['retained_months_of_revenue']} meses de su cuota. Medido en los "
               f"{len(ranked_customers):,} clientes de prueba. Cambia el porcentaje con el control de arriba.")

    st.markdown("#### Lista de prioridad para el equipo de retención")
    priority_list = (contacted_customers.merge(customer_data[["customerID"]], left_on="customer_index", right_index=True)
                     .assign(money_at_risk=lambda customers: customers["churn_probability"] * customers["MonthlyCharges"])
                     .sort_values("money_at_risk", ascending=False)
                     [["customerID", "money_at_risk", "churn_probability", "risk_segment", "MonthlyCharges", "tenure", "Contract"]]
                     .assign(Contract=lambda customers: customers["Contract"].map(display_value))  # display only
                     .rename(columns={"customerID": "ID de cliente", "money_at_risk": "dinero en riesgo / mes",
                                      "churn_probability": "riesgo de abandono", "risk_segment": "segmento",
                                      "MonthlyCharges": "cuota mensual", "tenure": "meses como cliente", "Contract": "contrato"}))
    st.caption("Ordenada por dinero en riesgo = probabilidad de abandono × cuota mensual: llama primero a los clientes "
               "cuya pérdida cuesta más, no solo a los más propensos a irse.")
    st.dataframe(priority_list.style.format({"dinero en riesgo / mes": "${:,.2f}", "riesgo de abandono": "{:.0%}",
                                             "cuota mensual": "${:,.2f}"}),
                 hide_index=True, width="stretch", height=300)
    st.download_button("Descargar lista de prioridad (CSV)", priority_list.to_csv(index=False),
                       "menolyte_lista_prioridad.csv", "text/csv")

    st.markdown("#### Próximamente: sistema multiagente de Menolyte")
    roadmap_columns = st.columns(3)
    roadmap_columns[0].info("**🧭 Agente orquestador**\n\nCoordina el ciclo mensual: evalúa la base, activa al agente "
                            "analista y al de campañas, y reúne los resultados.")
    roadmap_columns[1].info("**📈 Agente analista**\n\nLee predicciones, segmentos y explicaciones SHAP, y escribe "
                            "recomendaciones para el gerente en lenguaje simple.")
    roadmap_columns[2].info("**💬 Agente de campañas (WhatsApp)**\n\nContacta a los clientes de riesgo alto aprobados con "
                            "ofertas personalizadas y registra el resultado.")
    st.caption("Humano en el circuito: el gerente aprueba cada campaña. Los clientes se contactan solo con consentimiento "
               "previo, conforme a la Ley Orgánica de Protección de Datos Personales del Ecuador (LOPDP).")

# ----------------------------------------------------------------------------------------------------------------------
# Deferred: form-size curve of the Model Performance tab (slowest computation, cached after the first run).
# Drawn last so the Prediction and Business tabs appear immediately; it lands in the slot reserved in that tab.
# ----------------------------------------------------------------------------------------------------------------------
with form_curve_slot:
    curve = form_size_curve()
    chosen_form = curve.iloc[len(FORM_FIELDS) - 1]
    figure = px.line(curve, x="fields asked", y=["F2", "ROC-AUC"], markers=True,
                     color_discrete_map={"F2": "#2d2d2d", "ROC-AUC": NEUTRAL_COLOR},
                     labels={"fields asked": "datos pedidos", "value": "puntaje", "variable": "métrica"},
                     title=f"Con {len(FORM_FIELDS)} datos el modelo rinde casi igual que con los 19")
    figure.add_vline(x=len(FORM_FIELDS), line_dash="dot", line_color=HIGHLIGHT_COLOR,
                     annotation_text="Formulario de la app", annotation_position="bottom right")
    figure.update_layout(xaxis_title="Datos pedidos (los más importantes primero)", yaxis_title="Puntaje en prueba",
                         legend_title="", xaxis_dtick=1)
    st.plotly_chart(figure, width="stretch", theme=None)
    st.caption(f"Orden: {', '.join(PLAIN_LANGUAGE_LABELS[variable].lower() for variable in FIELD_RANKING)}. Los datos no pedidos toman el valor típico del entrenamiento; el total "
               f"pagado se calcula como meses × cuota mensual. El formulario de la app pide {len(FORM_FIELDS)} datos "
               f"(línea punteada): F2 {chosen_form['F2']:.3f} vs. {metadata['test_metrics']['f2']:.3f} con los 19, ROC-AUC "
               f"{chosen_form['ROC-AUC']:.3f} vs. {metadata['test_metrics']['roc_auc']:.3f}, y la misma decisión para el "
               f"{chosen_form['same decision as all fields']:.1%} de los clientes de prueba. El modelo no se reentrena.")
