"""Menolyte Core visual identity: global CSS and the Plotly chart template.

Presentation only: nothing here touches data, models or calculations.
Colors match .streamlit/config.toml so Streamlit widgets and Plotly charts look like one product.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BRAND_RED = "#d62728"
TEXT_DARK = "#2d2d2d"
NEUTRAL_GREY = "#b0b7c3"
MID_GREY = "#6b7280"
LIGHT_SURFACE = "#f5f6f8"
BORDER = "#e3e6eb"
GRID = "#eef0f3"
FONT_FAMILY = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Order matters: series without an explicit color take these in turn.
# Dark grey first (neutral "baseline"), brand red second (the one we want the eye on), then greys.
BRAND_COLORWAY = [TEXT_DARK, BRAND_RED, NEUTRAL_GREY, MID_GREY, "#f19c9c", "#8c1c1d"]

GLOBAL_CSS = f"""
<style>
/* ---------- Page: generous whitespace, content not glued to the top ---------- */
.block-container {{ padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1400px; }}

/* ---------- Hide Streamlit's developer chrome during the demo ---------- */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], footer {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- KPI cards: every st.metric becomes a card ---------- */
[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-left: 4px solid {BRAND_RED};
    border-radius: 0.75rem;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 4px 12px rgba(16, 24, 40, 0.06);
    height: 100%;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.85rem; font-weight: 600; color: {MID_GREY};
    text-transform: uppercase; letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{ font-weight: 700; color: {TEXT_DARK}; line-height: 1.15; }}
[data-testid="stMetricDelta"] {{ font-weight: 500; }}

/* ---------- Tabs: bigger, readable from the back of the room ---------- */
[data-baseweb="tab-list"] {{ gap: 0.5rem; border-bottom: 1px solid {BORDER}; }}
[data-baseweb="tab"] {{ padding: 0.6rem 1rem; }}
[data-baseweb="tab"] p {{ font-size: 1.05rem; font-weight: 600; }}

/* ---------- Section rhythm: clear breathing room before each section title ---------- */
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{ margin-top: 1.75rem; color: {TEXT_DARK}; }}
[data-testid="stHeadingWithActionElements"] h3 {{ margin-top: 0.5rem; }}

/* ---------- Charts sit on white cards like the KPIs ---------- */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {BORDER}; border-radius: 0.75rem; padding: 0.5rem; background: #ffffff;
}}
</style>
"""


def apply_global_styles():
    """Inject the global CSS. Call once, right after st.set_page_config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def register_plotly_template():
    """Register the 'menolyte' Plotly template and make it the default for every figure."""
    axis_style = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="",
                      title_font=dict(size=14, color=MID_GREY), tickfont=dict(size=13, color=TEXT_DARK), automargin=True)
    pio.templates["menolyte"] = go.layout.Template(layout=dict(
        font=dict(family=FONT_FAMILY, size=14, color=TEXT_DARK),
        title=dict(font=dict(size=18, color=TEXT_DARK), x=0, xanchor="left", y=0.97, yanchor="top"),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        colorway=BRAND_COLORWAY,
        xaxis={**axis_style, "showgrid": False},  # one grid direction is enough: less ink, same information
        yaxis=axis_style,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13), title_font=dict(size=13, color=MID_GREY)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(family=FONT_FAMILY, size=13, color=TEXT_DARK)),
        margin=dict(t=70, l=20, r=20, b=20),
        bargap=0.25,
    ))
    pio.templates.default = "plotly_white+menolyte"
