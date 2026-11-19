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
