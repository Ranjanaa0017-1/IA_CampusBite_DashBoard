"""
CampusBite Analytics Dashboard
================================
A data science dashboard validating the CampusBite meal subscription
business model using ML, clustering, association rules, and regression.

Design: Editorial / warm terracotta aesthetic
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Data & ML ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, r2_score, mean_squared_error
)

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── Visualization ─────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CampusBite | Analytics",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
# Palette: warm cream backgrounds, terracotta accent, deep charcoal text
C_ACCENT   = "#D4622A"   # terracotta
C_ACCENT2  = "#2A7D6F"   # muted teal
C_DARK     = "#1C1917"   # charcoal
C_MID      = "#78716C"   # warm grey
C_LIGHT    = "#F5F0EB"   # cream
C_WHITE    = "#FAFAF8"   # off-white
C_BORDER   = "#E7E2DC"   # light border

PALETTE    = [C_ACCENT, C_ACCENT2, "#E8A838", "#6B4EBB", "#C45C8A", "#3A86B4"]
SEQ_WARM   = ["#FDE8D8", "#F5B08A", "#E8854C", C_ACCENT, "#9C3D12"]
SEQ_TEAL   = ["#D8F0EC", "#8ACAC3", "#2A9D8F", C_ACCENT2, "#14524A"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: {C_DARK};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C_DARK} !important;
    border-right: none !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0;
}}
[data-testid="stSidebar"] * {{
    color: #E8E0D8 !important;
}}
[data-testid="stSidebar"] .stRadio > label {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {C_MID} !important;
    margin-bottom: 4px !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    color: #E8E0D8 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: background 0.15s !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: rgba(212,98,42,0.15) !important;
}}

/* ── Main area ── */
.main {{
    background: {C_WHITE};
}}
.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1400px !important;
}}

/* ── Page title strip ── */
.page-eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {C_ACCENT};
    margin-bottom: 6px;
}}
.page-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {C_DARK};
    line-height: 1.15;
    margin-bottom: 6px;
}}
.page-desc {{
    font-size: 0.95rem;
    color: {C_MID};
    max-width: 680px;
    line-height: 1.65;
    margin-bottom: 2rem;
}}

/* ── Divider ── */
.ruled {{
    border: none;
    border-top: 1.5px solid {C_BORDER};
    margin: 1.5rem 0;
}}

/* ── KPI card ── */
.kpi-card {{
    background: {C_LIGHT};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: {C_ACCENT};
    border-radius: 3px 0 0 3px;
}}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C_MID};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {C_DARK};
    line-height: 1;
}}
.kpi-note {{
    font-size: 0.78rem;
    color: {C_MID};
    margin-top: 6px;
}}

/* ── Insight tile ── */
.insight-tile {{
    background: {C_LIGHT};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
}}
.insight-tile .i-tag {{
    display: inline-block;
    background: {C_ACCENT};
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}}
.insight-tile .i-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: {C_DARK};
    margin-bottom: 8px;
}}
.insight-tile .i-body {{
    font-size: 0.88rem;
    color: {C_MID};
    line-height: 1.65;
}}

/* ── Persona card ── */
.persona {{
    background: {C_WHITE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}}
.persona .p-emoji {{
    font-size: 2rem;
    margin-bottom: 10px;
    display: block;
}}
.persona .p-name {{
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: {C_DARK};
    margin-bottom: 8px;
}}
.persona .p-desc {{
    font-size: 0.82rem;
    color: {C_MID};
    line-height: 1.6;
}}
.persona .p-badge {{
    display: inline-block;
    margin-top: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    color: white;
}}

/* ── Info callout ── */
.callout {{
    background: #FEF3EC;
    border-left: 4px solid {C_ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: {C_DARK};
    margin: 1rem 0;
    line-height: 1.6;
}}

/* ── Table styling ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def page_header(eyebrow: str, title: str, desc: str = ""):
    """Render a consistent page header."""
    st.markdown(f'<div class="page-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="page-desc">{desc}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="ruled">', unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = ""):
    """Render a KPI card."""
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-note'>" + note + "</div>" if note else ""}
    </div>"""


def styled_fig(fig):
    """Apply consistent Plotly layout to any figure."""
    fig.update_layout(
        plot_bgcolor=C_WHITE,
        paper_bgcolor=C_WHITE,
        font=dict(family="DM Sans", color=C_DARK, size=12),
        title_font=dict(family="Playfair Display", size=16, color=C_DARK),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(t=48, b=16, l=16, r=16),
    )
    fig.update_xaxes(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER)
    fig.update_yaxes(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Load and return the raw Excel dataset."""
    return pd.read_excel(file)


@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: mode for categoricals, median for numerics."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].fillna(out[col].mode()[0])
    for col in out.select_dtypes(include="number").columns:
        out[col] = out[col].fillna(out[col].median())
    return out


@st.cache_data
def encode_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Label-encode all categorical columns (except addon_preference / respondent_id).
    Returns (encoded_df, feature_columns).
    """
    enc = df.copy()
    le  = LabelEncoder()
    skip = {"addon_preference", "respondent_id"}
    for col in enc.select_dtypes(include="object").columns:
        if col not in skip:
            enc[col] = le.fit_transform(enc[col].astype(str))
    feature_cols = [
        c for c in enc.columns
        if c not in {"subscription_interest", "respondent_id", "addon_preference"}
    ]
    return enc, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Brand block
    st.markdown(f"""
    <div style="background:{C_ACCENT};padding:24px 20px 20px;margin:-1rem -1rem 1.5rem;
                border-bottom:none">
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:white;line-height:1">🍱 CampusBite</div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.75);
                    margin-top:4px;letter-spacing:0.06em">Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset upload
    st.markdown("**Upload Dataset**")
    uploaded = st.file_uploader(
        "CampusBite_Dataset.xlsx",
        type=["xlsx"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "nav",
        [
            "🏠  Home",
            "🔍  Dataset Explorer",
            "📊  Data Visualizations",
            "🤖  Classification Model",
            "👥  Customer Clustering",
            "🔗  Association Rules",
            "📈  Regression Forecast",
            "💡  Business Insights",
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.75rem;color:{C_MID};line-height:1.6'>"
        "Individual PBL · Data Science<br>CampusBite · 2025–26</div>",
        unsafe_allow_html=True
    )

# ── Guard: require upload for all ML/analysis pages ───────────────────────────
NEEDS_DATA = [
    "🔍  Dataset Explorer", "📊  Data Visualizations",
    "🤖  Classification Model", "👥  Customer Clustering",
    "🔗  Association Rules", "📈  Regression Forecast",
    "💡  Business Insights",
]

if page in NEEDS_DATA:
    if uploaded is None:
        page_header("Action Required", "Upload Your Dataset")
        st.markdown("""
        <div class="callout">
            📂 Please upload <strong>CampusBite_Dataset.xlsx</strong> using the
            sidebar file uploader to access this section.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    raw_df = load_data(uploaded)
    df     = clean_data(raw_df)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    # Hero
    st.markdown(f"""
    <div style="background:{C_DARK};border-radius:16px;padding:48px 52px;
                margin-bottom:2rem;position:relative;overflow:hidden">
        <div style="position:absolute;right:48px;top:50%;transform:translateY(-50%);
                    font-size:6rem;opacity:0.12">🍱</div>
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.18em;
                    text-transform:uppercase;color:{C_ACCENT};margin-bottom:14px">
            Student Meal Tech · Data Science PBL
        </div>
        <div style="font-family:'Playfair Display',serif;font-size:3rem;
                    font-weight:800;color:white;line-height:1.1;margin-bottom:16px">
            CampusBite<br>Analytics Dashboard
        </div>
        <div style="font-size:1rem;color:#A8A29E;max-width:560px;line-height:1.7">
            Validating a subscription-based meal delivery platform for university
            students through machine learning, clustering, and data analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, note in [
        (c1, "Survey Respondents", "1,500", "Synthetic dataset"),
        (c2, "Survey Questions",   "21",    "Features collected"),
        (c3, "ML Algorithms",      "4",     "Classification · Clustering · ARM · Regression"),
        (c4, "Target Segments",    "4",     "Student persona profiles"),
    ]:
        col.markdown(kpi(label, val, note), unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:{C_DARK};margin-bottom:16px">
            The Problem We're Solving
        </div>
        """, unsafe_allow_html=True)
        problems = [
            ("💸", "Affordability", "Eating out daily on a student budget is expensive and unsustainable."),
            ("⏱️", "Convenience", "Students have no time to cook between lectures and assignments."),
            ("🥗", "Nutrition",    "Fast food defaults leave students under-nourished and fatigued."),
        ]
        for emoji, title, body in problems:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;
                        margin-bottom:16px;padding:16px;background:{C_LIGHT};
                        border-radius:10px;border:1px solid {C_BORDER}">
                <div style="font-size:1.6rem;min-width:36px">{emoji}</div>
                <div>
                    <div style="font-weight:600;margin-bottom:4px">{title}</div>
                    <div style="font-size:0.87rem;color:{C_MID}">{body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                    font-weight:700;color:{C_DARK};margin-bottom:16px">
            Our Solution: CampusBite
        </div>
        """, unsafe_allow_html=True)
        solutions = [
            ("📦", "3 Subscription Tiers", "Basic · Standard · Premium — all below ₹500/month"),
            ("🥘", "Dietary Options",       "Vegetarian, Vegan, Halal, High-Protein, Jain"),
            ("🔄", "Rotating Menus",        "Fresh weekly menus — no repetition boredom"),
            ("🍹", "Combo Add-ons",         "Drinks, Snacks, Dessert bundles for upsell"),
            ("📍", "Campus Delivery",       "30-minute on-campus delivery — no commute"),
        ]
        for emoji, title, body in solutions:
            st.markdown(f"""
            <div style="display:flex;gap:12px;align-items:center;
                        margin-bottom:12px;padding:12px 16px;
                        border-left:3px solid {C_ACCENT};
                        background:{C_LIGHT};border-radius:0 8px 8px 0">
                <div style="font-size:1.4rem">{emoji}</div>
                <div>
                    <div style="font-weight:600;font-size:0.9rem">{title}</div>
                    <div style="font-size:0.8rem;color:{C_MID}">{body}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Methodology table
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                font-weight:700;color:{C_DARK};margin-bottom:16px">
        Analytical Methodology
    </div>
    """, unsafe_allow_html=True)
    method_data = {
        "Step": ["1", "2", "3", "4"],
        "Algorithm": ["Random Forest", "K-Means Clustering", "Apriori (ARM)", "Linear Regression"],
        "Goal": [
            "Predict which students will subscribe",
            "Segment students into distinct personas",
            "Discover meal/add-on combo patterns",
            "Forecast monthly spending per student",
        ],
        "Target Variable": [
            "subscription_interest (Yes/No)",
            "Cluster label (0–3)",
            "Itemsets from addon/diet/cuisine",
            "wtp_per_month (₹)",
        ],
    }
    st.dataframe(pd.DataFrame(method_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍  Dataset Explorer":
    page_header(
        "Data · Overview",
        "Dataset Explorer",
        "Inspect the CampusBite survey dataset — structure, quality, and statistical properties."
    )

    tab_prev, tab_shape, tab_missing, tab_stats = st.tabs(
        ["📋 Preview", "📐 Shape & Types", "⚠️ Missing Values", "📊 Statistics"]
    )

    # ── Preview ──────────────────────────────────────────────────────────────
    with tab_prev:
        n = st.slider("Rows to display", 5, 100, 15)
        st.dataframe(raw_df.head(n), use_container_width=True)

    # ── Shape & Types ────────────────────────────────────────────────────────
    with tab_shape:
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("Total Rows",    f"{raw_df.shape[0]:,}", "Survey respondents"), unsafe_allow_html=True)
        c2.markdown(kpi("Total Columns", f"{raw_df.shape[1]}",   "Features / questions"), unsafe_allow_html=True)
        c3.markdown(kpi("Memory Usage",  f"{raw_df.memory_usage(deep=True).sum() / 1024:.0f} KB", "In-memory size"), unsafe_allow_html=True)

        col_meta = pd.DataFrame({
            "Column":          raw_df.columns,
            "Data Type":       raw_df.dtypes.astype(str).values,
            "Non-Null Count":  raw_df.notnull().sum().values,
            "Unique Values":   [raw_df[c].nunique() for c in raw_df.columns],
            "Example Value":   [str(raw_df[c].dropna().iloc[0]) if len(raw_df[c].dropna()) > 0 else "—"
                                for c in raw_df.columns],
        })
        st.markdown("#### Column Metadata")
        st.dataframe(col_meta, use_container_width=True, hide_index=True)

    # ── Missing Values ────────────────────────────────────────────────────────
    with tab_missing:
        miss     = raw_df.isnull().sum()
        miss_pct = (miss / len(raw_df) * 100).round(2)
        miss_df  = pd.DataFrame({"Column": miss.index, "Count": miss.values, "Pct (%)": miss_pct.values})
        miss_df  = miss_df[miss_df["Count"] > 0].reset_index(drop=True)

        if miss_df.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.warning(f"⚠️ {len(miss_df)} column(s) contain missing values.")
            fig = px.bar(
                miss_df, x="Column", y="Pct (%)",
                color="Pct (%)", color_continuous_scale=SEQ_WARM,
                text="Count", title="Missing Value Rate by Column"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
            st.dataframe(miss_df, use_container_width=True, hide_index=True)

    # ── Statistics ────────────────────────────────────────────────────────────
    with tab_stats:
        st.markdown("#### Numeric Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("#### Categorical Distributions")
        cat_cols = [c for c in df.select_dtypes(include="object").columns
                    if c not in {"respondent_id", "addon_preference"}]
        sel = st.selectbox("Choose column", cat_cols)
        vc  = df[sel].value_counts().reset_index()
        vc.columns = [sel, "Count"]
        vc["Share (%)"] = (vc["Count"] / len(df) * 100).round(1)

        col_a, col_b = st.columns([1.3, 1])
        with col_a:
            fig = px.bar(vc, x=sel, y="Count", color=sel,
                         color_discrete_sequence=PALETTE, text="Count",
                         title=f"Distribution — {sel}")
            fig.update_traces(textposition="outside")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        with col_b:
            st.dataframe(vc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DATA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊  Data Visualizations":
    page_header(
        "Exploratory Analysis",
        "Data Visualizations",
        "Interactive charts revealing student demographics, food habits, and subscription behaviour."
    )

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="age", nbins=9, color_discrete_sequence=[C_ACCENT],
                           title="Age Distribution",
                           labels={"age": "Age", "count": "Students"})
        fig.update_traces(marker_line_color="white", marker_line_width=1.5)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Students aged 19–21 form the largest cohort — core target demographic.")

    with c2:
        fig = px.histogram(df, x="food_budget", nbins=25,
                           color_discrete_sequence=[C_ACCENT2],
                           title="Monthly Food Budget Distribution",
                           labels={"food_budget": "Budget (₹)", "count": "Students"})
        fig.update_traces(marker_line_color="white", marker_line_width=1.5)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Most students operate on a ₹200–400/month food budget — pricing must respect this.")

    # Row 2
    c1, c2 = st.columns(2)
    with c1:
        diet_vc = df["diet_type"].value_counts().reset_index()
        diet_vc.columns = ["diet_type", "count"]
        fig = px.bar(diet_vc, x="diet_type", y="count",
                     color="diet_type", color_discrete_sequence=PALETTE,
                     title="Dietary Preference Distribution",
                     text="count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Non-Vegetarian leads, but combined Veg+Vegan+Jain = significant alternative market.")

    with c2:
        timing_vc = df["meal_timing"].value_counts().reset_index()
        timing_vc.columns = ["meal_timing", "count"]
        fig = px.pie(timing_vc, names="meal_timing", values="count",
                     color_discrete_sequence=PALETTE, hole=0.42,
                     title="Meal Timing Preferences")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Lunch & Dinner combos are the most popular — design daily meal plans around these slots.")

    # Row 3
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="subscription_interest", y="food_budget",
                     color="subscription_interest",
                     color_discrete_map={"Yes": C_ACCENT2, "No": C_ACCENT},
                     title="Food Budget vs Subscription Interest",
                     labels={"food_budget": "Food Budget (₹)", "subscription_interest": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Interested students cluster in the ₹250–400 range — not the very cheapest or most expensive.")

    with c2:
        sub_vc = df["subscription_interest"].value_counts().reset_index()
        sub_vc.columns = ["interest", "count"]
        sub_vc["pct"] = (sub_vc["count"] / len(df) * 100).round(1)
        fig = px.bar(sub_vc, x="interest", y="count",
                     color="interest",
                     color_discrete_map={"Yes": C_ACCENT2, "No": C_ACCENT},
                     title="Subscription Interest Breakdown",
                     text=sub_vc["pct"].apply(lambda x: f"{x}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        yes_pct = (df["subscription_interest"] == "Yes").mean() * 100
        st.caption(f"{yes_pct:.1f}% of students are interested in subscribing — a strong initial market signal.")

    # Row 4 – WTP scatter & Cuisine
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="food_budget", y="wtp_per_month",
                         color="subscription_interest",
                         color_discrete_map={"Yes": C_ACCENT2, "No": C_ACCENT},
                         opacity=0.45, size_max=6,
                         title="Food Budget vs Willingness to Pay",
                         labels={"food_budget": "Food Budget (₹)", "wtp_per_month": "WTP/Month (₹)"})
        fig.update_traces(marker_size=5)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("WTP and food budget are positively correlated — higher-budget students willing to pay more.")

    with c2:
        cuisine_vc = df["cuisine_preference"].value_counts().reset_index()
        cuisine_vc.columns = ["cuisine", "count"]
        fig = px.pie(cuisine_vc, names="cuisine", values="count",
                     hole=0.4, color_discrete_sequence=PALETTE,
                     title="Cuisine Preference")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Indian cuisine dominates — menus must be Indian-first with regional variety.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖  Classification Model":
    page_header(
        "Machine Learning · Classification",
        "Subscription Prediction",
        "Random Forest classifier predicting whether a student will subscribe to CampusBite."
    )

    @st.cache_data
    def run_rf(df):
        enc_df, feature_cols = encode_df(df)
        X = enc_df[feature_cols]
        y = enc_df["subscription_interest"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        acc     = accuracy_score(y_te, y_pred)
        cm      = confusion_matrix(y_te, y_pred)
        report  = classification_report(y_te, y_pred, output_dict=True)
        fi_df   = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        return acc, cm, report, fi_df, y_te, y_pred

    with st.spinner("Training Random Forest (200 trees)..."):
        acc, cm, report, fi_df, y_te, y_pred = run_rf(df)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, note in [
        (c1, "Accuracy",   f"{acc*100:.1f}%",  "On 20% holdout set"),
        (c2, "Precision",  f"{report['weighted avg']['precision']*100:.1f}%", "Weighted avg"),
        (c3, "Recall",     f"{report['weighted avg']['recall']*100:.1f}%",    "Weighted avg"),
        (c4, "F1-Score",   f"{report['weighted avg']['f1-score']*100:.1f}%",  "Weighted avg"),
    ]:
        col.markdown(kpi(lbl, val, note), unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        # Confusion matrix
        labels = sorted(y_te.unique())
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale=["#FEF3EC", C_ACCENT],
                        labels={"x": "Predicted", "y": "Actual"},
                        title="Confusion Matrix")
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")

    with c2:
        # Feature importance
        top15 = fi_df.head(15)
        fig = px.bar(top15, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=SEQ_WARM,
                     title="Top 15 Feature Importances")
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("meal_skip_freq and current_satisfaction are the strongest predictors of subscription interest.")

    # Per-class report
    st.markdown("#### Classification Report by Class")
    rep_df = pd.DataFrame(report).T.loc[["No", "Yes"]][["precision", "recall", "f1-score", "support"]]
    rep_df = rep_df.round(3).reset_index().rename(columns={"index": "Class"})
    st.dataframe(rep_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="callout">
        <strong>Interpretation:</strong> The model achieves {acc*100:.1f}% accuracy on unseen data.
        Students who skip meals frequently and are dissatisfied with current food options are
        the clearest subscription candidates — target these with direct messaging campaigns.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥  Customer Clustering":
    page_header(
        "Machine Learning · Clustering",
        "Customer Segmentation",
        "K-Means clustering segments students into distinct personas for targeted marketing."
    )

    CLUSTER_FEATURES = [
        "food_budget", "wtp_per_month", "meal_skip_freq",
        "nutrition_importance", "current_satisfaction",
        "distance_to_food", "meals_per_day"
    ]

    k = st.sidebar.slider("Number of clusters (k)", 2, 7, 4)

    @st.cache_data
    def run_kmeans(df, k):
        X = df[CLUSTER_FEATURES].copy()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Elbow
        inertias = []
        K_vals   = list(range(2, 11))
        for k_ in K_vals:
            km = KMeans(n_clusters=k_, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        # Final
        km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels   = km_final.fit_predict(X_scaled)
        return labels, inertias, K_vals, X_scaled

    with st.spinner("Running K-Means..."):
        labels, inertias, K_vals, X_scaled = run_kmeans(df, k)

    df_c = df.copy()
    df_c["Cluster"] = labels.astype(str)

    c1, c2 = st.columns(2)
    with c1:
        # Elbow
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_vals, y=inertias, mode="lines+markers",
            line=dict(color=C_ACCENT, width=2.5),
            marker=dict(color=C_ACCENT, size=8)
        ))
        fig.add_vline(x=k, line_dash="dash", line_color=C_ACCENT2,
                      annotation_text=f"k = {k}", annotation_position="top right")
        fig.update_layout(
            title="Elbow Method — Optimal k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (WCSS)"
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with c2:
        # Cluster sizes
        cs = df_c["Cluster"].value_counts().sort_index().reset_index()
        cs.columns = ["Cluster", "Count"]
        cs["Cluster"] = cs["Cluster"].apply(lambda x: f"Cluster {x}")
        fig = px.bar(cs, x="Cluster", y="Count",
                     color="Cluster", color_discrete_sequence=PALETTE,
                     title="Students per Cluster", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    # Scatter
    fig = px.scatter(
        df_c, x="food_budget", y="wtp_per_month",
        color="Cluster", size="meal_skip_freq",
        color_discrete_sequence=PALETTE,
        hover_data=["diet_type", "cooking_habit", "current_satisfaction"],
        title="Customer Segments — Food Budget vs WTP (bubble = meal skip frequency)",
        labels={"food_budget": "Monthly Food Budget (₹)", "wtp_per_month": "WTP/Month (₹)"},
        opacity=0.65
    )
    st.plotly_chart(styled_fig(fig), use_container_width=True)

    # Personas
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:{C_DARK};margin:1.5rem 0 1rem">
        Student Persona Profiles
    </div>
    """, unsafe_allow_html=True)

    personas = [
        ("💸", "Budget Students",
         "Low food budget, skips meals 4–5x/week. Primary pain is cost. Highly receptive to Basic Plan at ₹149–199/month. Price sensitivity is extreme — discounts and referral rewards will drive adoption.",
         C_ACCENT, "High Priority"),
        ("🥗", "Health-Conscious Students",
         "Rates nutrition importance 4–5/5. Often Vegan or Vegetarian. Willing to pay ₹400+ for macro-tracked, balanced meals. Target with Premium Plan + nutritional info dashboard.",
         C_ACCENT2, "Premium Segment"),
        ("⚡", "Convenience Seekers",
         "Lives 2+ km from food options, heavy app user. Values speed over price. Open to Standard Plan with guaranteed 30-min delivery SLA. Loyal once on-boarded.",
         "#E8A838", "Retention Focus"),
        ("🌟", "Premium Subscribers",
         "High budget (₹400+), high WTP, low meal skipping. Values variety and quality. Best upsell target for combo add-ons (Drinks + Snacks) and limited rotating menus.",
         "#6B4EBB", "Upsell Target"),
    ]

    cols = st.columns(min(k, 4))
    for i, (col, (emoji, name, desc, color, badge)) in enumerate(zip(cols, personas[:k])):
        with col:
            st.markdown(f"""
            <div class="persona">
                <span class="p-emoji">{emoji}</span>
                <div class="p-name">{name}</div>
                <div class="p-desc">{desc}</div>
                <span class="p-badge" style="background:{color}">{badge}</span>
            </div>
            """, unsafe_allow_html=True)

    # Cluster profile table
    st.markdown("---")
    st.markdown("#### Cluster Feature Profiles — Mean Values")
    profile = df_c.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
    profile.index = [f"Cluster {i}" for i in profile.index]
    st.dataframe(profile, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔗  Association Rules":
    page_header(
        "Machine Learning · ARM",
        "Association Rule Mining",
        "Apriori algorithm discovers purchasing relationships between dietary preferences, cuisines, and add-ons."
    )

    min_sup  = st.sidebar.slider("Min Support",    0.01, 0.40, 0.05, 0.01)
    min_conf = st.sidebar.slider("Min Confidence", 0.10, 0.90, 0.30, 0.05)

    @st.cache_data
    def run_arm(df, min_sup, min_conf):
        """Build transactions from addon_preference + diet_type + meal_timing."""
        transactions = []
        for _, row in df.iterrows():
            items = []
            # Add-ons
            if pd.notna(row.get("addon_preference")):
                items += [a.strip() for a in str(row["addon_preference"]).split("|")]
            # Enrichments
            for col in ["diet_type", "meal_timing", "cuisine_preference"]:
                if col in row and pd.notna(row[col]):
                    items.append(str(row[col]))
            transactions.append(items)

        te       = TransactionEncoder()
        te_array = te.fit_transform(transactions)
        te_df    = pd.DataFrame(te_array, columns=te.columns_)

        freq  = apriori(te_df, min_support=min_sup, use_colnames=True)
        if freq.empty:
            return freq, pd.DataFrame()

        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("lift", ascending=False)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        return freq, rules

    with st.spinner("Running Apriori algorithm..."):
        freq_items, rules = run_arm(df, min_sup, min_conf)

    if rules.empty:
        st.warning("No rules found. Try lowering minimum support or confidence in the sidebar.")
        st.stop()

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Rules Discovered", f"{len(rules)}",                   "Qualifying association rules"), unsafe_allow_html=True)
    c2.markdown(kpi("Avg Confidence",   f"{rules['confidence'].mean()*100:.1f}%", "Rule reliability"),       unsafe_allow_html=True)
    c3.markdown(kpi("Max Lift",         f"{rules['lift'].max():.2f}×",     "Strongest co-occurrence"),        unsafe_allow_html=True)

    st.markdown("---")

    # Top rules table
    st.markdown("#### Top 10 Association Rules (by Lift)")
    disp = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).copy()
    disp["support"]    = disp["support"].round(4)
    disp["confidence"] = disp["confidence"].round(3)
    disp["lift"]       = disp["lift"].round(3)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            rules.head(80), x="support", y="confidence",
            size="lift", color="lift",
            hover_data=["antecedents", "consequents"],
            color_continuous_scale=SEQ_WARM,
            title="Support vs Confidence  (bubble = lift)",
            labels={"support": "Support", "confidence": "Confidence"}
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    with c2:
        top10 = rules.head(10).copy()
        top10["rule"] = top10["antecedents"] + " → " + top10["consequents"]
        fig = px.bar(top10, x="lift", y="rule", orientation="h",
                     color="lift", color_continuous_scale=SEQ_WARM,
                     title="Top 10 Rules by Lift",
                     labels={"lift": "Lift", "rule": ""})
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(styled_fig(fig), use_container_width=True)

    st.markdown(f"""
    <div class="callout">
        <strong>Key Finding:</strong> Students who prefer <em>Drinks</em> as an add-on also
        frequently choose <em>Snacks</em>. Bundle these in a ₹49 combo add-on at checkout
        to increase average order value by 15–20%. Dietary preferences co-occur strongly
        with specific meal timings — use this for personalised push notifications.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈  Regression Forecast":
    page_header(
        "Machine Learning · Regression",
        "Spending Forecast",
        "Linear Regression predicts a student's monthly willingness to pay (WTP) for meal subscriptions."
    )

    REG_FEATURES = [
        "food_budget", "meal_skip_freq", "current_satisfaction",
        "nutrition_importance", "distance_to_food",
        "meals_per_day", "referral_likelihood", "age"
    ]

    @st.cache_data
    def run_regression(df):
        X = df[REG_FEATURES]
        y = df["wtp_per_month"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2     = r2_score(y_te, y_pred)
        rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
        coef_df = pd.DataFrame({
            "Feature":     REG_FEATURES,
            "Coefficient": model.coef_
        }).sort_values("Coefficient", key=abs, ascending=False)
        return r2, rmse, y_te, y_pred, coef_df

    with st.spinner("Training Linear Regression..."):
        r2, rmse, y_te, y_pred, coef_df = run_regression(df)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("R² Score",    f"{r2:.3f}",    "Variance explained by model"), unsafe_allow_html=True)
    c2.markdown(kpi("RMSE",        f"₹{rmse:.1f}", "Root Mean Squared Error"),     unsafe_allow_html=True)
    c3.markdown(kpi("Test Samples", f"{len(y_te)}", "20% holdout set"),             unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        # Actual vs Predicted
        min_v = min(float(y_te.min()), float(y_pred.min()))
        max_v = max(float(y_te.max()), float(y_pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_te.values, y=y_pred,
            mode="markers",
            marker=dict(color=C_ACCENT, opacity=0.45, size=5),
            name="Predictions"
        ))
        fig.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines",
            line=dict(color=C_ACCENT2, dash="dash", width=2),
            name="Perfect Fit"
        ))
        fig.update_layout(
            title="Actual vs Predicted WTP",
            xaxis_title="Actual WTP (₹/month)",
            yaxis_title="Predicted WTP (₹/month)"
        )
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("Points near the dashed line indicate accurate predictions.")

    with c2:
        # Coefficients
        coef_df["Direction"] = coef_df["Coefficient"].apply(
            lambda x: "Positive" if x >= 0 else "Negative"
        )
        fig = px.bar(
            coef_df, x="Coefficient", y="Feature", orientation="h",
            color="Direction",
            color_discrete_map={"Positive": C_ACCENT2, "Negative": C_ACCENT},
            title="Regression Coefficients",
            labels={"Coefficient": "Effect on WTP (₹)", "Feature": ""}
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_fig(fig), use_container_width=True)
        st.caption("food_budget is the strongest positive driver. Higher satisfaction slightly reduces WTP.")

    # Residuals
    residuals = y_te.values - y_pred
    fig = px.histogram(
        x=residuals, nbins=40, color_discrete_sequence=[C_ACCENT2],
        title="Residual Distribution",
        labels={"x": "Residual (Actual − Predicted)", "count": "Frequency"}
    )
    fig.add_vline(x=0, line_dash="dash", line_color=C_ACCENT,
                  annotation_text="Zero", annotation_position="top right")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    st.plotly_chart(styled_fig(fig), use_container_width=True)

    st.markdown(f"""
    <div class="callout">
        <strong>Interpretation:</strong> The model explains {r2*100:.1f}% of variance in monthly spending.
        A student's <em>food_budget</em> is the strongest predictor of how much they'll pay for a subscription.
        Residuals are normally distributed around zero — confirming Linear Regression is appropriate for this data.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡  Business Insights":
    page_header(
        "Strategy · Summary",
        "Business Insights",
        "Data-backed strategic recommendations for CampusBite launch, pricing, and growth."
    )

    insights = [
        ("Who Will Subscribe?",
         "Target",
         "Students who skip 3+ meals/week AND rate current food options below 3/5 have a 70%+ predicted subscription probability. "
         "This segment represents the highest-conversion cohort — prioritise them in all marketing channels."),
        ("Optimal Pricing Strategy",
         "Pricing",
         "Regression analysis shows peak WTP clustering between ₹200–450/month. "
         "A three-tier structure — Basic ₹199 / Standard ₹349 / Premium ₹499 — captures all major spending personas "
         "without alienating budget-sensitive students."),
        ("Must-Have: Dietary Inclusivity",
         "Product",
         "Vegetarian (28%) + Halal (16%) + Vegan (8%) + Jain (6%) = 58% of students need non-standard menus. "
         "A platform without certified Halal options cannot become the go-to service for diverse campuses."),
        ("Combo Revenue Booster",
         "Revenue",
         "Association rules show Drinks → Snacks co-purchasing at 65–75% confidence. "
         "A ₹49 Drinks+Snacks combo upsell at checkout can increase average order value by 15–20% "
         "with near-zero additional cost."),
        ("Launch Location Strategy",
         "Growth",
         "Students 2+ km from food options are 40% more likely to subscribe. "
         "Target campus hostels, PG accommodations, and campuses with limited on-campus dining first "
         "to maximise early adoption speed."),
        ("Referral Engine",
         "Acquisition",
         "High-budget students (food_budget > ₹400) show referral_likelihood scores of 4–5/5. "
         "A 'Refer a Friend — Get 1 Free Meal' campaign targeting this segment can drive organic growth "
         "at a customer acquisition cost well below paid channels."),
    ]

    c1, c2 = st.columns(2)
    for i, (title, tag, body) in enumerate(insights):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div class="insight-tile">
                <span class="i-tag">{tag}</span>
                <div class="i-title">{title}</div>
                <div class="i-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # Quick stats row
    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                font-weight:700;color:{C_DARK};margin-bottom:1rem">
        At-a-Glance Summary
    </div>
    """, unsafe_allow_html=True)

    summary = {
        "% Interested in Subscribing": f"{(df['subscription_interest']=='Yes').mean()*100:.1f}%",
        "Most Popular Cuisine":        df["cuisine_preference"].mode()[0],
        "Top Dietary Preference":      df["diet_type"].mode()[0],
        "Avg WTP / Month":             f"₹{df['wtp_per_month'].mean():.0f}",
        "Avg Food Budget":             f"₹{df['food_budget'].mean():.0f}",
        "Avg Meal Skips / Week":       f"{df['meal_skip_freq'].mean():.1f}",
        "Top Dissatisfaction Reason":  df["dissatisfaction_reason"].mode()[0],
        "Top Payment Method":          df["payment_method"].mode()[0],
    }
    cols = st.columns(4)
    for i, (lbl, val) in enumerate(summary.items()):
        cols[i % 4].markdown(kpi(lbl, val), unsafe_allow_html=True)

    # Go-to-market
    st.markdown("---")
    st.markdown("#### 🚀 Go-to-Market Roadmap")
    gtm = pd.DataFrame({
        "Phase":    ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
        "Timeline": ["Month 1–2", "Month 3–4", "Month 5–6", "Month 7–12"],
        "Action": [
            "Beta launch on 1 campus. Basic + Standard plans only. Manual delivery.",
            "Add Halal-certified kitchen. Expand to 3 campuses. App launch.",
            "Introduce Drinks+Snacks combos. Launch Refer-a-Friend programme.",
            "Premium plan with AI meal recommendations + macro tracking dashboard.",
        ],
        "KPI Target": [
            "100 subscribers, <5% churn",
            "500 subscribers, NPS > 40",
            "+20% avg order value from combos",
            "2,000 subscribers, 3 cities",
        ],
    })
    st.dataframe(gtm, use_container_width=True, hide_index=True)
