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
/* The header only held those controls: hide it so the sticky tab bar can use the top edge */
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stHeader"], footer {{ display: none !important; }}

/* ---------- KPI cards: every st.metric becomes a card ---------- */
[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-left: 4px solid {NEUTRAL_GREY};  /* neutral: red is reserved for risk (deltas, charts, primary action) */
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
/* Cards in the same row share the same height, even when one label wraps to two lines */
[data-testid="stColumn"]:has([data-testid="stMetric"]) > [data-testid="stVerticalBlock"],
[data-testid="stColumn"] [data-testid="stElementContainer"]:has([data-testid="stMetric"]),
[data-testid="stColumn"] [data-testid="stElementContainer"]:has([data-testid="stMetric"]) > div {{ height: 100%; }}
/* Labels and deltas wrap onto a second line instead of being cut with "…" */
[data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"] p,
[data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"], [data-testid="stMetricDelta"] p {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.3;
}}

/* ---------- Filter chips: dark grey instead of red (red = risk only) ---------- */
span[data-baseweb="tag"] {{ background-color: {TEXT_DARK} !important; }}

/* ---------- Analyst tools hidden during the demo: Plotly modebar, heading link icons ---------- */
[data-testid="stPlotlyChart"] .modebar-container, [data-testid="stHeaderActionElements"] {{ display: none !important; }}

/* ---------- Tabs: sticky navigation bar, bigger text readable from the back of the room ---------- */
/* Streamlit 1.60 markup: [data-testid="stTabs"] > div > ([role="tablist"] > [data-testid="stTab"]) + [role="tabpanel"] */
[data-testid="stTabs"] [role="tablist"] {{
    position: sticky; top: 0; z-index: 100;           /* stays pinned while stMain scrolls */
    background: rgba(255, 255, 255, 0.96);            /* opaque enough that content never shows through */
    backdrop-filter: blur(6px);
    gap: 0.5rem; border-bottom: 1px solid {BORDER};
    padding-top: 0.5rem;
    box-shadow: 0 6px 12px -10px rgba(16, 24, 40, 0.25);  /* subtle edge so it reads as "floating" over content */
}}
[data-testid="stTab"] {{ padding: 0.6rem 1rem; }}
[data-testid="stTab"] p {{ font-size: 1.05rem; font-weight: 600; color: {MID_GREY}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {TEXT_DARK}; }}

/* ---------- Section rhythm: clear breathing room before each section title ---------- */
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {{ margin-top: 1.75rem; color: {TEXT_DARK}; }}
[data-testid="stHeadingWithActionElements"] h3 {{ margin-top: 0.5rem; }}

/* ---------- Brand header (top of every page) ---------- */
.mn-brand {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem; }}
.mn-logo {{
    width: 2.6rem; height: 2.6rem; border-radius: 0.7rem; background: {BRAND_RED}; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.4rem;
    box-shadow: 0 4px 12px rgba(214, 39, 40, 0.25);
}}
.mn-name {{ font-size: 1.6rem; font-weight: 800; color: {TEXT_DARK}; letter-spacing: -0.02em; line-height: 1.1; }}
.mn-name span {{ color: {BRAND_RED}; }}
.mn-tagline {{ color: {MID_GREY}; font-size: 0.95rem; }}

/* ---------- Home: hero, big figure cards, how-to steps ---------- */
.mn-hero {{ padding: 2.5rem 0 1.5rem 0; }}
.mn-eyebrow {{ color: {BRAND_RED}; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.85rem; }}
.mn-headline {{
    font-size: 2.7rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.03em;
    color: {TEXT_DARK}; max-width: 900px; margin: 0.5rem 0 0.75rem 0;
}}
.mn-headline span {{ color: {BRAND_RED}; }}
.mn-subhead {{ font-size: 1.15rem; color: {MID_GREY}; max-width: 780px; }}
.mn-card {{
    background: #ffffff; border: 1px solid {BORDER}; border-top: 5px solid {BRAND_RED}; border-radius: 0.9rem;
    padding: 1.6rem 1.5rem; height: 100%;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.07);
}}
.mn-card-value {{ font-size: 3.4rem; font-weight: 800; color: {TEXT_DARK}; line-height: 1; letter-spacing: -0.03em; }}
.mn-card-title {{ font-size: 1.1rem; font-weight: 600; color: {TEXT_DARK}; margin-top: 0.75rem; }}
.mn-card-detail {{ font-size: 0.92rem; color: {MID_GREY}; margin-top: 0.4rem; }}
.mn-step {{ background: {LIGHT_SURFACE}; border-radius: 0.9rem; padding: 1.25rem; height: 100%; }}
.mn-step-number {{
    width: 2rem; height: 2rem; border-radius: 50%; background: {TEXT_DARK}; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 0.75rem;
}}
.mn-step-title {{ font-weight: 700; font-size: 1.05rem; color: {TEXT_DARK}; }}
.mn-step-tab {{ font-size: 0.85rem; color: {BRAND_RED}; font-weight: 600; margin: 0.15rem 0 0.4rem 0; }}
.mn-step-text {{ font-size: 0.95rem; color: {MID_GREY}; }}
.mn-section-title {{ font-size: 1.35rem; font-weight: 700; color: {TEXT_DARK}; margin: 2.5rem 0 1rem 0; }}

/* ---------- Prediction result card: top border colored by risk segment (st.container key = prediction_card_<0|1|2>) ---------- */
[class*="st-key-prediction_card_"] {{
    border-radius: 0.9rem !important; padding: 0.75rem 1.25rem !important; background: #ffffff;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.07);
}}
.st-key-prediction_card_0 {{ border-top: 6px solid {NEUTRAL_GREY} !important; }}
.st-key-prediction_card_1 {{ border-top: 6px solid #f19c9c !important; }}
.st-key-prediction_card_2 {{ border-top: 6px solid {BRAND_RED} !important; }}
[class*="st-key-prediction_card_"] [data-testid="stPlotlyChart"] {{ border: none; padding: 0; }}
.mn-badge {{ display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 1rem; }}
.mn-badge-0 {{ background: #eef0f3; color: {TEXT_DARK}; }}
.mn-badge-1 {{ background: #fbe3e3; color: #8c1c1d; }}
.mn-badge-2 {{ background: {BRAND_RED}; color: #ffffff; }}
.mn-result-label {{ font-size: 0.85rem; font-weight: 600; color: {MID_GREY}; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 1.4rem; }}
.mn-result-decision {{ font-size: 2rem; font-weight: 800; color: {TEXT_DARK}; letter-spacing: -0.02em; margin-top: 0.2rem; }}
.mn-result-plain {{ font-size: 1.05rem; color: {MID_GREY}; margin-top: 0.4rem; }}

/* ---------- Retention actions: numbered priority cards ---------- */
.mn-action {{
    display: flex; gap: 0.9rem; align-items: flex-start; background: #ffffff; border: 1px solid {BORDER};
    border-radius: 0.8rem; padding: 1rem 1.1rem; height: 100%;
}}
.mn-action-number {{
    flex: 0 0 2rem; height: 2rem; border-radius: 50%; background: {BRAND_RED}; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
}}
.mn-action-text {{ font-weight: 600; color: {TEXT_DARK}; line-height: 1.35; }}
.mn-action-reason {{ font-size: 0.88rem; color: {MID_GREY}; margin-top: 0.35rem; }}

/* ---------- Charts sit on white cards like the KPIs ---------- */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {BORDER}; border-radius: 0.75rem; padding: 0.5rem; background: #ffffff;
}}
/* Streamlit 1.60 injects the theme's secondary background as plot_bgcolor; keep the plot area white */
[data-testid="stPlotlyChart"] rect.bg {{ fill: #ffffff !important; }}
</style>
"""


def apply_global_styles():
    """Inject the global CSS. Call once, right after st.set_page_config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def html(markup):
    """Render trusted, app-authored HTML (brand header, Home cards)."""
    st.markdown(markup, unsafe_allow_html=True)


def brand_header(tagline):
    return (f'<div class="mn-brand"><div class="mn-logo">M</div><div>'
            f'<div class="mn-name">Menolyte <span>Core</span></div><div class="mn-tagline">{tagline}</div></div></div>')


def figure_card(value, title, detail):
    return (f'<div class="mn-card"><div class="mn-card-value">{value}</div>'
            f'<div class="mn-card-title">{title}</div><div class="mn-card-detail">{detail}</div></div>')


def step_card(number, title, tab_name, text):
    return (f'<div class="mn-step"><div class="mn-step-number">{number}</div><div class="mn-step-title">{title}</div>'
            f'<div class="mn-step-tab">→ {tab_name}</div><div class="mn-step-text">{text}</div></div>')


def risk_badge(text, level):
    """Pill with the risk segment; level 0 = low (grey), 1 = medium (light red), 2 = high (red)."""
    return f'<span class="mn-badge mn-badge-{level}">{text}</span>'


def action_card(number, action, reason):
    return (f'<div class="mn-action"><div class="mn-action-number">{number}</div><div>'
            f'<div class="mn-action-text">{action}</div><div class="mn-action-reason">{reason}</div></div></div>')


def risk_gauge(probability, segment_edges, level):
    """Gauge 0–100% with the risk segments as background bands; the bar takes the segment color."""
    bar_color = [NEUTRAL_GREY, "#f19c9c", BRAND_RED][level]
    edges = [edge * 100 for edge in segment_edges]
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number={"suffix": "%", "font": {"size": 56, "color": TEXT_DARK}, "valueformat": ".0f"},
        title={"text": "Probabilidad de abandono", "font": {"size": 15, "color": MID_GREY}},
        gauge={"axis": {"range": [0, 100], "tickvals": edges, "ticktext": [f"{edge:.0f}%" for edge in edges],
                        "tickfont": {"size": 12, "color": MID_GREY}},
               "bar": {"color": bar_color, "thickness": 0.35}, "bgcolor": "#ffffff", "borderwidth": 0,
               "steps": [{"range": [edges[0], edges[1]], "color": "#f3f4f6"},
                         {"range": [edges[1], edges[2]], "color": "#fdf0f0"},
                         {"range": [edges[2], edges[3]], "color": "#fbe3e3"}]}))
    figure.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30))
    return figure


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
        # Horizontal legend above the plot: never steals width from the chart, long names are not cut
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13), title_font=dict(size=13, color=MID_GREY),
                    orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(family=FONT_FAMILY, size=13, color=TEXT_DARK)),
        margin=dict(t=70, l=20, r=36, b=20),  # right margin: room for the last tick label (e.g. "100%")
        bargap=0.25,
    ))
    pio.templates.default = "plotly_white+menolyte"
