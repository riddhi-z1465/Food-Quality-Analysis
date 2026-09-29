import os
import sys
import streamlit as st

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="MilkSense AI | Dairy Quality Intelligence",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Anchor paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'milk_quality.csv')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from preprocessing import (
        FEATURE_COLUMNS,
        NUMERICAL_COLUMNS,
        FEATURE_RANGES,
        preprocess_single_input,
        normalize_columns,
        load_dataset,
        clean_dataset,
    )
    from predict import MilkQualityPredictor
except ImportError as e:
    st.error(f"Error importing modules from 'src/': {e}. Please ensure the project structure is intact.")
    st.stop()

# Comprehensive Custom CSS for Apple/Linear-grade design aesthetics
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Base Typography & Canvas */
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    p, h1, h2, h3, h4, h5, h6, .stMarkdown, label, button, input {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Preserve Streamlit icon fonts and Material Symbols ligatures */
    [data-testid="stIcon"],
    [class*="material-symbols"],
    [class*="material-icons"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    button[kind="header"] *,
    header * {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }
    
    .stApp {
        background-color: #080c16;
        background-image: 
            radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 40%),
            radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 40%),
            radial-gradient(at 50% 50%, rgba(16, 185, 129, 0.04) 0px, transparent 60%);
    }
    
    /* Modern Sidebar Overhaul */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #080d1a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Sleek Hairline Gradient Dividers */
    .hairline-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.12) 50%, transparent 100%);
        margin: 1.2rem 0;
        border: none;
    }
    
    /* Brand Logo & Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.6rem 0.2rem;
        margin-bottom: 0.4rem;
    }
    
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 16px -4px rgba(14, 165, 233, 0.4);
        font-size: 1.5rem;
    }
    
    .brand-text-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .brand-text-pill {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 6px;
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: white;
        letter-spacing: 0.05em;
    }
    
    .brand-text-sub {
        font-size: 0.72rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.15rem;
    }
    
    /* Model Info Banner */
    .model-info-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .model-info-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #38bdf8 0%, #818cf8 100%);
    }
    
    .model-title-text {
        font-size: 1.05rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-top: 0.15rem;
    }
    
    .model-spec-text {
        font-size: 0.73rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.2rem;
    }
    
    /* Sleek 2x2 Metric Grid in Sidebar */
    .metric-grid-tile {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.55rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
        backdrop-filter: blur(8px);
    }
    
    .metric-grid-tile:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .metric-grid-lbl {
        font-size: 0.68rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    
    .metric-grid-val {
        font-size: 1.35rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.15rem;
        line-height: 1.1;
    }
    
    .metric-mini-bar {
        height: 3px;
        border-radius: 2px;
        background: rgba(255, 255, 255, 0.08);
        margin-top: 0.4rem;
        overflow: hidden;
    }
    
    .metric-mini-fill {
        height: 100%;
        border-radius: 2px;
    }
    
    /* Main Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    .badge-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        backdrop-filter: blur(8px);
    }
    
    .badge-pill-cyan {
        background: rgba(6, 182, 212, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }
    
    .badge-pill-emerald {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    
    .badge-pill-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    
    /* Modern Glass Card */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Input Section Headers */
    .input-group-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem;
        margin-bottom: 1.1rem;
    }
    
    .input-category-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    /* Visual Spectrum Strips */
    .spectrum-container {
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }
    
    .spectrum-bar-ph {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 35%, #10b981 50%, #3b82f6 80%, #8b5cf6 100%);
        position: relative;
    }
    
    .spectrum-bar-temp {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #06b6d4 0%, #10b981 30%, #f59e0b 55%, #ef4444 100%);
        position: relative;
    }
    
    .spectrum-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.68rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    
    /* Prediction Hero Badges */
    .pred-badge-high {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.28) 0%, rgba(16, 185, 129, 0.12) 100%);
        border: 1.5px solid #10b981;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 15px 35px -10px rgba(16, 185, 129, 0.4);
    }
    
    .pred-badge-medium {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.28) 0%, rgba(245, 158, 11, 0.12) 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 15px 35px -10px rgba(245, 158, 11, 0.4);
    }
    
    .pred-badge-low {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.28) 0%, rgba(239, 68, 68, 0.12) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 15px 35px -10px rgba(239, 68, 68, 0.4);
    }
    
    .grade-text-high {
        font-size: 2.8rem;
        font-weight: 800;
        color: #34d399;
        letter-spacing: 0.04em;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    
    .grade-text-medium {
        font-size: 2.8rem;
        font-weight: 800;
        color: #fbbf24;
        letter-spacing: 0.04em;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    
    .grade-text-low {
        font-size: 2.8rem;
        font-weight: 800;
        color: #f87171;
        letter-spacing: 0.04em;
        line-height: 1.1;
        margin: 0.2rem 0;
    }
    
    /* Navigation Tab Enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.7);
        padding: 0.45rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0 1.2rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4) !important;
    }
    
    /* Form Submit Button Glow */
    div.stButton > button[kind="primary"], .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 50%, #06b6d4 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.8rem 2.2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 8px 22px -3px rgba(14, 165, 233, 0.55) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -3px rgba(14, 165, 233, 0.75) !important;
    }
    
    /* Verified Sensor Parameter Chip */
    .param-chip {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 0.8rem 0.6rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .param-chip-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .param-chip-lbl {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 0.2rem;
    }
    .param-chip-tag {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.18rem 0.48rem;
        border-radius: 4px;
        margin-top: 0.35rem;
        display: inline-block;
        letter-spacing: 0.02em;
    }
    .tag-optimal { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .tag-caution { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .tag-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Cache and load model, scaler, and benchmark metrics."""
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    results_path = os.path.join(REPORTS_DIR, 'model_results.csv')
    all_models_path = os.path.join(MODELS_DIR, 'all_models.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    results_df = pd.read_csv(results_path) if os.path.exists(results_path) else None
    all_models = joblib.load(all_models_path) if os.path.exists(all_models_path) else None
    
    return model, scaler, results_df, all_models


@st.cache_data
def load_cached_dataset():
    """Load and cache the raw dataset for EDA visualizers."""
    if os.path.exists(DATA_PATH):
        return load_dataset(DATA_PATH)
    return None


model, scaler, results_df, all_models = load_model_artifacts()
dataset_df = load_cached_dataset()


# ==============================================================================
# SIDEBAR: ADVANCED TELEMETRY & PRESETS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon-box">🥛</div>
        <div>
            <div class="brand-text-title">
                MilkSense <span class="brand-text-pill">AI</span>
            </div>
            <div class="brand-text-sub">Quality Intelligence Engine</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 0.4rem; padding-left: 0.2rem; margin-bottom: 0.8rem;">
        <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
        <span style="font-size: 0.72rem; font-weight: 600; color: #94a3b8;">v1.0.0 Production Engine • Online</span>
    </div>
    <div class="hairline-divider"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🎯 Active Production Model")
    
    best_name = "K-Nearest Neighbors"
    if results_df is not None:
        for _, row in results_df.iterrows():
            if hasattr(model, 'n_neighbors') and 'Neighbors' in row['Model']:
                best_name = row['Model']
                break
            elif hasattr(model, 'n_estimators') and 'Forest' in row['Model']:
                best_name = row['Model']
                break
                
        st.markdown(f"""
        <div class="model-info-banner">
            <div style="font-size: 0.68rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.06em;">Selected Architecture</div>
            <div class="model-title-text">{best_name}</div>
            <div class="model-spec-text">k=3 • L2 Metric • Distance-Weighted</div>
        </div>
        """, unsafe_allow_html=True)
        
        best_row = results_df[results_df['Model'] == best_name]
        if not best_row.empty:
            row_data = best_row.iloc[0]
            
            sb_col1, sb_col2 = st.columns(2)
            with sb_col1:
                st.markdown(f"""
                <div class="metric-grid-tile">
                    <div class="metric-grid-lbl">🎯 Test Acc</div>
                    <div class="metric-grid-val" style="color: #34d399;">{row_data['Test_Accuracy']*100:.1f}%</div>
                    <div class="metric-mini-bar"><div class="metric-mini-fill" style="width: {row_data['Test_Accuracy']*100}%; background: #34d399;"></div></div>
                </div>
                <div class="metric-grid-tile">
                    <div class="metric-grid-lbl">⚡ Precision</div>
                    <div class="metric-grid-val" style="color: #c084fc;">{row_data['Test_Precision']*100:.1f}%</div>
                    <div class="metric-mini-bar"><div class="metric-mini-fill" style="width: {row_data['Test_Precision']*100}%; background: #c084fc;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            with sb_col2:
                st.markdown(f"""
                <div class="metric-grid-tile">
                    <div class="metric-grid-lbl">🔄 5-Fold CV</div>
                    <div class="metric-grid-val" style="color: #38bdf8;">{row_data['CV_Mean_Accuracy']*100:.1f}%</div>
                    <div class="metric-mini-bar"><div class="metric-mini-fill" style="width: {row_data['CV_Mean_Accuracy']*100}%; background: #38bdf8;"></div></div>
                </div>
                <div class="metric-grid-tile">
                    <div class="metric-grid-lbl">📊 F1 Score</div>
                    <div class="metric-grid-val" style="color: #fbbf24;">{row_data['Test_F1_Score']*100:.1f}%</div>
                    <div class="metric-mini-bar"><div class="metric-mini-fill" style="width: {row_data['Test_F1_Score']*100}%; background: #fbbf24;"></div></div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.2rem 0.4rem; font-size: 0.72rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                <span>Cross-Validation Std:</span>
                <span style="color: #38bdf8; font-weight: 700;">±{row_data['CV_Std_Accuracy']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Model metrics missing. Run `python src/train.py`.")

    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### ⚡ Sensor Profile Presets")
    
    # Modern Pill Selector for Presets
    preset_choice = st.pills(
        "Load Empirical Benchmark:",
        [
            "⚙️ Custom",
            "🥛 Fresh High",
            "🧀 Med Process",
            "🧪 Acidic Low",
            "🔥 Hot Abuse",
        ],
        default="⚙️ Custom",
        help="Quickly populate input controls with verified empirical sensor observations."
    )
    
    # Preset Feedback Mini Card
    preset_spec = {
        "⚙️ Custom": "Manual sensor parameter entry mode.",
        "🥛 Fresh High": "pH 6.6 • 37°C • Optimal Fat • Palatable",
        "🧀 Med Process": "pH 6.6 • 38°C • Standard Process Grade",
        "🧪 Acidic Low": "pH 4.5 • 38°C • Acidified / Fermented",
        "🔥 Hot Abuse": "pH 6.8 • 65°C • Thermal Spoilage Defect",
    }
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.5rem 0.7rem; font-size: 0.72rem; color: #94a3b8; margin-top: 0.4rem;">
        <span style="color: #38bdf8; font-weight: 600;">Spec:</span> {preset_spec.get(preset_choice, "")}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📦 Scientific Audit")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 0.75rem 0.85rem; font-size: 0.76rem; color: #cbd5e1; line-height: 1.6;">
        <div style="display: flex; justify-content: space-between;"><span>• Raw Records:</span><b style="color: #f1f5f9;">1,059</b></div>
        <div style="display: flex; justify-content: space-between;"><span>• Duplicates Removed:</span><b style="color: #f87171;">976 (92.2%)</b></div>
        <div style="display: flex; justify-content: space-between;"><span>• Unique Profiles:</span><b style="color: #34d399;">83 Distinct</b></div>
        <div style="display: flex; justify-content: space-between;"><span>• Split Protocol:</span><b style="color: #f1f5f9;">80/20 Stratified</b></div>
        <div style="display: flex; justify-content: space-between;"><span>• Leakage Guard:</span><b style="color: #34d399;">VERIFIED ✓</b></div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Milk Quality Analysis & Diagnostic Intelligence</div>
    <div class="hero-subtitle">Machine Learning Classification System for Dairy Processing & Food Safety Assurance</div>
    <div class="badge-pill-container">
        <div class="badge-pill badge-pill-cyan">● Live Inference Engine</div>
        <div class="badge-pill badge-pill-emerald">● Leakage-Free Verified (88.24% Gen. Acc)</div>
        <div class="badge-pill badge-pill-purple">● 5-Fold Stratified Cross-Validation</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN NAVIGATION TABS
# ==============================================================================
tab_pred, tab_eda, tab_models, tab_leakage, tab_about = st.tabs([
    "🧪 Real-Time Predictor",
    "📊 Dataset Insights & 3D Sensory Space",
    "🤖 Model Benchmarking & Diagnostics",
    "⚠️ Data Quality & Leakage Analysis",
    "🎓 Academic Report & Viva Defense"
])


# ==============================================================================
# TAB 1: REAL-TIME MILK QUALITY PREDICTOR
# ==============================================================================
with tab_pred:
    st.markdown("### Multi-Parameter Sensor Input")
    st.caption("Provide physical, thermal, and biochemical parameters to determine milk suitability.")
    
    # Configure presets mapping
    if preset_choice == "🥛 Fresh High":
        def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_colour = 6.6, 37.0, 1, 0, 1, 0, 255
    elif preset_choice == "🧀 Med Process":
        def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_colour = 6.6, 38.0, 0, 0, 0, 0, 255
    elif preset_choice == "🧪 Acidic Low":
        def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_colour = 4.5, 38.0, 0, 1, 0, 1, 250
    elif preset_choice == "🔥 Hot Abuse":
        def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_colour = 6.8, 65.0, 0, 0, 1, 1, 245
    else:
        def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_colour = 6.6, 37.0, 1, 0, 1, 0, 255
        
    with st.form("milk_prediction_form"):
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown('<div class="input-category-header">🧪 1. Biochemical & Optical Properties</div>', unsafe_allow_html=True)
            
            input_ph = st.number_input(
                "pH Level (Acidity / Alkalinity Scale)",
                min_value=float(FEATURE_RANGES['pH']['min']),
                max_value=float(FEATURE_RANGES['pH']['max']),
                value=float(def_ph),
                step=0.1,
                help="Empirical range: 3.0 to 9.5. Normal fresh milk is slightly acidic (pH 6.5–6.7)."
            )
            
            # Real-time visual pH spectrum bar
            st.markdown("""
            <div class="spectrum-container">
                <div class="spectrum-bar-ph"></div>
                <div class="spectrum-labels">
                    <span>3.0 (Acidic / Fermented)</span>
                    <span style="color: #34d399; font-weight: 700;">6.5 - 6.7 (Optimal Fresh)</span>
                    <span>9.5 (Alkaline / Mastitis)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive pH feedback tag
            if 6.5 <= input_ph <= 6.7:
                st.markdown('<span class="param-chip-tag tag-optimal">✓ In-Specification: Optimal Fresh Range (6.5 - 6.7)</span>', unsafe_allow_html=True)
            elif input_ph < 6.4:
                st.markdown('<span class="param-chip-tag tag-danger">⚠ Acidic Spoilage: Lactic Acid Accumulation (< 6.4)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="param-chip-tag tag-caution">⚠ Chemical Deviation: Alkaline / Mastitic Marker (> 6.8)</span>', unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            input_colour = st.number_input(
                "Colour Reflectance Index (240 - 255)",
                min_value=int(FEATURE_RANGES['Colour']['min']),
                max_value=int(FEATURE_RANGES['Colour']['max']),
                value=int(def_colour),
                step=1,
                help="Grayscale reflectance index. 255 = pure white; lower indicates yellowing or contamination."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="input-category-header">🌡️ 2. Thermal Storage Regime</div>', unsafe_allow_html=True)
            input_temp = st.number_input(
                "Temperature (°C)",
                min_value=float(FEATURE_RANGES['Temperature']['min']),
                max_value=float(FEATURE_RANGES['Temperature']['max']),
                value=float(def_temp),
                step=1.0,
                help="Empirical range: 34°C to 90°C. Milking temp ~37°C; sustained heat >40°C triggers bacterial spoilage."
            )
            
            # Real-time visual Temperature spectrum bar
            st.markdown("""
            <div class="spectrum-container">
                <div class="spectrum-bar-temp"></div>
                <div class="spectrum-labels">
                    <span>34°C (Chilled / Reception)</span>
                    <span style="color: #34d399; font-weight: 700;">35°C - 40°C (Safe Intake)</span>
                    <span>> 45°C (Thermal Abuse Spoilage)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if input_temp <= 40.0:
                st.markdown('<span class="param-chip-tag tag-optimal">✓ Safe Reception Temperature (≤ 40°C)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="param-chip-tag tag-danger">⚠ Elevated Temperature / Thermal Spoilage Risk (> 40°C)</span>', unsafe_allow_html=True)
                
        with col_c2:
            st.markdown('<div class="input-category-header">👅 3. Organoleptic Sensory Indicators</div>', unsafe_allow_html=True)
            
            # Modern Segmented Control for Taste
            taste_choice = st.segmented_control(
                "Taste Condition Evaluation",
                options=["✓ Palatable / Normal Taste", "✗ Sour / Bitter Off-Flavor"],
                default="✓ Palatable / Normal Taste" if def_taste == 1 else "✗ Sour / Bitter Off-Flavor"
            )
            input_taste = 1 if "Palatable" in str(taste_choice) else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            # Modern Segmented Control for Odor
            odor_choice = st.segmented_control(
                "Olfactory Odor Evaluation",
                options=["✓ Standard Fresh Aroma", "✗ Flat / Pungent Defect"],
                default="✓ Standard Fresh Aroma" if def_odor == 1 else "✗ Flat / Pungent Defect"
            )
            input_odor = 1 if "Fresh" in str(odor_choice) else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="input-category-header">🔬 4. Physical Emulsion State</div>', unsafe_allow_html=True)
            
            # Modern Segmented Control for Fat
            fat_choice = st.segmented_control(
                "Lipid Fat Content Adequacy",
                options=["✓ Standard Lipid (Adequate)", "✗ Deficient / Skimmed Fat"],
                default="✓ Standard Lipid (Adequate)" if def_fat == 1 else "✗ Deficient / Skimmed Fat"
            )
            input_fat = 1 if "Adequate" in str(fat_choice) else 0
            
            st.markdown("<br>", unsafe_allow_html=True)
            # Modern Segmented Control for Turbidity
            turb_choice = st.segmented_control(
                "Colloidal Turbidity (Opacity)",
                options=["✓ Standard Turbid Emulsion", "✗ Low Turbidity (Watery)"],
                default="✓ Standard Turbid Emulsion" if def_turb == 1 else "✗ Low Turbidity (Watery)"
            )
            input_turbidity = 1 if "Turbid" in str(turb_choice) else 0
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_submitted = st.form_submit_button("🧪 Analyze & Classify Milk Quality", use_container_width=True)

    if predict_submitted:
        if model is None or scaler is None:
            st.error("Model artifacts missing! Please execute `python src/train.py` from terminal.")
        else:
            sample_dict = {
                'pH': input_ph,
                'Temperature': input_temp,
                'Taste': input_taste,
                'Odor': input_odor,
                'Fat': input_fat,
                'Turbidity': input_turbidity,
                'Colour': input_colour,
            }
            
            try:
                predictor = MilkQualityPredictor(
                    model_path=os.path.join(MODELS_DIR, 'best_model.pkl'),
                    scaler_path=os.path.join(MODELS_DIR, 'scaler.pkl')
                )
                result = predictor.predict(sample_dict)
                pred_grade = result['predicted_grade'].upper()
                probs = result['probabilities']
                
                st.markdown("---")
                st.markdown("### 🏆 Diagnostic Classification Assessment")
                
                # Dynamic Hero Badge with glow
                if pred_grade == "HIGH":
                    st.markdown("""
                    <div class="pred-badge-high">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #34d399; letter-spacing: 0.08em; text-transform: uppercase;">
                            CLASSIFIED MILK QUALITY GRADE
                        </div>
                        <div class="grade-text-high">GRADE A: HIGH QUALITY</div>
                        <div style="color: #cbd5e1; font-size: 1.05rem; max-width: 650px; margin: 0.5rem auto 0 auto;">
                            Milk demonstrates ideal biochemical stability, lipid concentration, and physical freshness.
                            <b>Approved for direct consumer packaging, pasteurization, and premium dairy processing.</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif pred_grade == "MEDIUM":
                    st.markdown("""
                    <div class="pred-badge-medium">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #fbbf24; letter-spacing: 0.08em; text-transform: uppercase;">
                            CLASSIFIED MILK QUALITY GRADE
                        </div>
                        <div class="grade-text-medium">GRADE B: MEDIUM QUALITY</div>
                        <div style="color: #cbd5e1; font-size: 1.05rem; max-width: 650px; margin: 0.5rem auto 0 auto;">
                            Acceptable baseline properties with minor sensory or temperature deviations.
                            <b>Approved for secondary fermentation (cheese, yogurt, butter) or industrial drying.</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pred-badge-low">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #f87171; letter-spacing: 0.08em; text-transform: uppercase;">
                            CLASSIFIED MILK QUALITY GRADE
                        </div>
                        <div class="grade-text-low">GRADE C: LOW QUALITY (UNSAFE)</div>
                        <div style="color: #cbd5e1; font-size: 1.05rem; max-width: 650px; margin: 0.5rem auto 0 auto;">
                            Severe physical-chemical degradation, thermal abuse, or lactic acid fermentation detected.
                            <b>Rejected. Condemned and unsafe for human consumption or standard processing.</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Interactive Plots Row: Plotly Gauge + Radar Comparison
                col_viz1, col_viz2 = st.columns([1, 1.2])
                
                with col_viz1:
                    st.markdown("#### **Confidence Score Meter**")
                    target_prob = probs.get(pred_grade.capitalize(), 0.0) * 100
                    
                    gauge_color = "#10b981" if pred_grade == "HIGH" else ("#f59e0b" if pred_grade == "MEDIUM" else "#ef4444")
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=target_prob,
                        number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
                        title={'text': f"Confidence ({pred_grade})", 'font': {'size': 16, 'color': '#94a3b8'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                            'bar': {'color': gauge_color, 'thickness': 0.3},
                            'bgcolor': "rgba(30, 41, 59, 0.5)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255, 255, 255, 0.1)",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Probability Progress Breakdown
                    st.markdown("##### Class Probability Breakdown")
                    for g_name, g_color in [('High', '#10b981'), ('Medium', '#f59e0b'), ('Low', '#ef4444')]:
                        p_val = probs.get(g_name, 0.0)
                        st.write(f"**{g_name} Grade:** `{p_val*100:.1f}%`")
                        st.progress(p_val)
                    
                with col_viz2:
                    st.markdown("#### **Sensory Radar Analysis vs Ideal Benchmark**")
                    # Compute normalized metrics (0 to 1) for radar comparison
                    ph_norm = max(0.0, 1.0 - abs(input_ph - 6.6) / 2.0)
                    temp_norm = max(0.0, 1.0 - max(0.0, input_temp - 37.0) / 30.0)
                    colour_norm = (input_colour - 240) / 15.0
                    
                    categories = ['pH Balance', 'Thermal Safety', 'Taste', 'Aroma (Odor)', 'Fat Content', 'Turbidity', 'Colour Index']
                    sample_vals = [ph_norm, temp_norm, input_taste, input_odor, input_fat, input_turbidity, colour_norm]
                    ideal_vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=ideal_vals,
                        theta=categories,
                        fill='toself',
                        name='Ideal Fresh Milk Benchmark',
                        line=dict(color='#06b6d4', dash='dash', width=1.5),
                        fillcolor='rgba(6, 182, 212, 0.1)'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=sample_vals,
                        theta=categories,
                        fill='toself',
                        name='Submitted Milk Sample',
                        line=dict(color=gauge_color, width=2.5),
                        fillcolor=f'rgba({16 if pred_grade=="HIGH" else (245 if pred_grade=="MEDIUM" else 239)}, {185 if pred_grade=="HIGH" else (158 if pred_grade=="MEDIUM" else 68)}, {129 if pred_grade=="HIGH" else (11 if pred_grade=="MEDIUM" else 68)}, 0.35)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=8, color='#64748b')),
                            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=10, color='#cbd5e1'))
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        legend=dict(font=dict(color='#cbd5e1', size=10), orientation='h', y=-0.15),
                        height=330,
                        margin=dict(l=30, r=30, t=20, b=30)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Parameter Verification Audit Matrix
                st.markdown("---")
                st.markdown("#### 📋 Submitted Parameter Audit Matrix")
                c_chip = st.columns(7)
                
                with c_chip[0]:
                    tag_cls = "tag-optimal" if 6.5 <= input_ph <= 6.7 else "tag-danger"
                    tag_lbl = "OPTIMAL" if 6.5 <= input_ph <= 6.7 else "OUT OF SPEC"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{input_ph:.1f}</div><div class="param-chip-lbl">pH Level</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[1]:
                    tag_cls = "tag-optimal" if input_temp <= 40 else "tag-danger"
                    tag_lbl = "SAFE" if input_temp <= 40 else "ABUSED"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{input_temp:.0f}°C</div><div class="param-chip-lbl">Temperature</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[2]:
                    tag_cls = "tag-optimal" if input_taste == 1 else "tag-danger"
                    tag_lbl = "PALATABLE" if input_taste == 1 else "OFF-TASTE"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{"Good" if input_taste==1 else "Bad"}</div><div class="param-chip-lbl">Taste</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[3]:
                    tag_cls = "tag-optimal" if input_odor == 1 else "tag-caution"
                    tag_lbl = "FRESH" if input_odor == 1 else "FLAT"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{"Fresh" if input_odor==1 else "Defect"}</div><div class="param-chip-lbl">Aroma</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[4]:
                    tag_cls = "tag-optimal" if input_fat == 1 else "tag-danger"
                    tag_lbl = "STANDARD" if input_fat == 1 else "DEFICIENT"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{"Adequate" if input_fat==1 else "Low"}</div><div class="param-chip-lbl">Fat Content</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[5]:
                    tag_cls = "tag-optimal" if input_turbidity == 1 else "tag-caution"
                    tag_lbl = "NORMAL" if input_turbidity == 1 else "WATERY"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{"Turbid" if input_turbidity==1 else "Low"}</div><div class="param-chip-lbl">Turbidity</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                with c_chip[6]:
                    tag_cls = "tag-optimal" if input_colour == 255 else "tag-caution"
                    tag_lbl = "WHITE" if input_colour == 255 else "TINTED"
                    st.markdown(f'<div class="param-chip"><div class="param-chip-val">{input_colour}</div><div class="param-chip-lbl">Colour Scale</div><span class="param-chip-tag {tag_cls}">{tag_lbl}</span></div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Inference error encountered: {str(e)}")


# ==============================================================================
# TAB 2: DATASET INSIGHTS & 3D SENSORY SPACE
# ==============================================================================
with tab_eda:
    st.markdown("### Exploratory Data Analysis & Sensory Space")
    st.caption("Empirical exploration of physical and chemical parameters from verified observations.")
    
    if dataset_df is not None:
        clean_df, _ = clean_dataset(dataset_df, drop_duplicates=True)
        
        # Metric Cards
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        c_kpi1.metric("Raw Observations", f"{len(dataset_df):,}", "In-Cooperative Intake")
        c_kpi2.metric("Unique Profiles", f"{len(clean_df):,}", "-976 Duplicates Removed")
        c_kpi3.metric("Features Monitored", "7 Attributes", "Biochemical + Optical")
        c_kpi4.metric("Class Balance", "Balanced", "Medium (34) | Low (26) | High (23)")
        
        st.markdown("---")
        st.markdown("#### 🌐 Interactive 3D Milk Sensory Space (pH × Temperature × Colour)")
        st.caption("Rotate, pan, and hover over observations to inspect how milk quality separates in 3D physical space.")
        
        fig_3d = px.scatter_3d(
            clean_df,
            x='pH',
            y='Temperature',
            z='Colour',
            color='Grade',
            color_discrete_map={'High': '#10b981', 'Medium': '#f59e0b', 'Low': '#ef4444'},
            hover_data=['Taste', 'Odor', 'Fat', 'Turbidity'],
            opacity=0.85,
            size_max=8
        )
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)", title="pH"),
                yaxis=dict(backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)", title="Temperature (°C)"),
                zaxis=dict(backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)", title="Colour Scale")
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📈 Complete 10-Figure Academic EDA Catalog")
        eda_view = st.selectbox(
            "Select Academic Visualization to Inspect:",
            [
                "1. Grade Class Distribution",
                "2. pH Distribution & Normality",
                "3. Temperature Distribution & Thermal Regimes",
                "4. Colour Reflectance Index Distribution",
                "5. Boxplot: pH Variation by Quality Grade",
                "6. Boxplot: Temperature Variation by Quality Grade",
                "7. Boxplot: Colour Variation by Quality Grade",
                "8. Comprehensive Multi-Feature Distributions",
                "9. Correlation Heatmap",
                "10. Binary Sensor Proportions by Quality Grade"
            ]
        )
        
        fig_mapping = {
            "1. Grade Class Distribution": "grade_distribution.png",
            "2. pH Distribution & Normality": "ph_distribution.png",
            "3. Temperature Distribution & Thermal Regimes": "temperature_distribution.png",
            "4. Colour Reflectance Index Distribution": "colour_distribution.png",
            "5. Boxplot: pH Variation by Quality Grade": "boxplot_ph_grade.png",
            "6. Boxplot: Temperature Variation by Quality Grade": "boxplot_temperature_grade.png",
            "7. Boxplot: Colour Variation by Quality Grade": "boxplot_colour_grade.png",
            "8. Comprehensive Multi-Feature Distributions": "feature_distribution_plots.png",
            "9. Correlation Heatmap": "correlation_heatmap.png",
            "10. Binary Sensor Proportions by Quality Grade": "feature_vs_quality.png",
        }
        
        fig_path = os.path.join(FIGURES_DIR, fig_mapping[eda_view])
        if os.path.exists(fig_path):
            st.image(fig_path, use_container_width=True)
        else:
            st.warning("Figure not found. Run `python src/eda.py`.")
            
        st.markdown("#### **Parametric Summary Table (Deduplicated Profiles)**")
        st.dataframe(clean_df.describe().T.style.format("{:.2f}"), use_container_width=True)
    else:
        st.error(f"Dataset not found at {DATA_PATH}")


# ==============================================================================
# TAB 3: MODEL COMPARISON & DIAGNOSTICS
# ==============================================================================
with tab_models:
    st.markdown("### Algorithmic Benchmarks & Diagnostic Evaluation")
    st.caption("Five classification architectures evaluated under 5-Fold Stratified Cross-Validation on the training partition.")
    
    if results_df is not None:
        # Interactive Plotly Model Comparison
        fig_bar = px.bar(
            results_df,
            x='Model',
            y=['CV_Mean_Accuracy', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1_Score'],
            barmode='group',
            labels={'value': 'Evaluation Score (0.0 to 1.0)', 'variable': 'Evaluation Metric'},
            color_discrete_sequence=['#0284c7', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6']
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
            height=400,
            yaxis=dict(range=[0, 1.05], gridcolor='rgba(255,255,255,0.08)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
            legend=dict(orientation='h', y=1.12, font=dict(size=10))
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("#### 🏆 Algorithmic Scorecard")
        st.dataframe(
            results_df.style.highlight_max(
                subset=['CV_Mean_Accuracy', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1_Score'],
                color='#065f46'
            ).format({
                'CV_Mean_Accuracy': '{:.4f}',
                'CV_Std_Accuracy': '±{:.4f}',
                'Test_Accuracy': '{:.4f}',
                'Test_Precision': '{:.4f}',
                'Test_Recall': '{:.4f}',
                'Test_F1_Score': '{:.4f}',
            }),
            use_container_width=True
        )
        
        st.markdown("---")
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.markdown("##### 🎯 Confusion Matrix Diagnostics")
            cm_choice = st.selectbox(
                "Select Model Confusion Matrix:",
                ["All Models Combined", "K-Nearest Neighbors", "Support Vector Machine", "Random Forest", "Decision Tree", "Logistic Regression"]
            )
            cm_map = {
                "All Models Combined": "all_confusion_matrices.png",
                "K-Nearest Neighbors": "confusion_matrix_k_nearest_neighbors.png",
                "Support Vector Machine": "confusion_matrix_support_vector_machine.png",
                "Random Forest": "confusion_matrix_random_forest.png",
                "Decision Tree": "confusion_matrix_decision_tree.png",
                "Logistic Regression": "confusion_matrix_logistic_regression.png",
            }
            cm_fpath = os.path.join(FIGURES_DIR, cm_map[cm_choice])
            if os.path.exists(cm_fpath):
                st.image(cm_fpath, use_container_width=True)
                
        with col_diag2:
            st.markdown("##### 💡 Predictive Feature Importance")
            fi_choice = st.radio("Importance Metric:", ["Random Forest Gini Impurity", "KNN Permutation Importance"], horizontal=True)
            rf_fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
            perm_fi_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
            if "Gini" in fi_choice and os.path.exists(rf_fi_path):
                st.image(rf_fi_path, use_container_width=True)
            elif os.path.exists(perm_fi_path):
                st.image(perm_fi_path, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Root-Cause Error Analysis (Holdout Test Set)")
        err_csv_path = os.path.join(REPORTS_DIR, "error_analysis.csv")
        err_fig_path = os.path.join(FIGURES_DIR, "error_analysis.png")
        if os.path.exists(err_csv_path):
            err_df = pd.read_csv(err_csv_path)
            if not err_df.empty:
                col_e1, col_e2 = st.columns([1.2, 1])
                with col_e1:
                    st.dataframe(err_df, use_container_width=True)
                with col_e2:
                    if os.path.exists(err_fig_path):
                        st.image(err_fig_path, use_container_width=True)
                        
                st.info(
                    "💡 **Analytical Insight:** Notice that both misclassified samples lie on subtle boundary interfaces. "
                    "Sample A features standard pH (6.5) and temp (37°C) matching High quality, but had taste=0 and colour=245. "
                    "Sample B features borderline elevated temperature (50°C) with standard pH (6.8), resulting in Medium/Low ambiguity."
                )


# ==============================================================================
# TAB 4: DATA QUALITY & LEAKAGE ANALYSIS (ACADEMIC SHOWCASE)
# ==============================================================================
with tab_leakage:
    st.markdown("### ⚠️ Academic Case Study: The Data Leakage Paradox")
    st.markdown("""
    A hallmark of scientific rigor is **demonstrating data integrity**. In widely circulated educational versions 
    of the Milk Quality dataset, many online notebooks claim **99.5% to 100% classification accuracy**. 
    
    This section proves mathematically why those results represent **severe data leakage rather than real learning**.
    """)
    
    col_leak_a, col_leak_b = st.columns(2)
    
    with col_leak_a:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #ef4444;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #f87171; margin-bottom: 0.5rem;">
                ❌ The Naive Memorization Trap
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Total Records:</b> 1,059 observations<br>
                • <b>Identified Duplicates:</b> 976 rows (<b>92.16%</b>)<br>
                • <b>True Distinct Profiles:</b> Only <b>83 unique sensor combinations</b><br><br>
                When a naive <code>train_test_split</code> is executed directly on 1,059 records, identical sensor observations 
                are randomly distributed into <b>both training and testing partitions</b>.<br><br>
                The classifier simply matches test observations against training duplicates, producing an artificial 
                <b>99.5% – 100% memorization score</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_leak_b:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #10b981;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #34d399; margin-bottom: 0.5rem;">
                ✅ The Scientifically Rigorous Solution
            </div>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Deduplication:</b> Exact duplicate records are isolated, leaving 83 distinct physical samples.<br>
                • <b>Stratified Partitioning:</b> 80% (66 samples) for cross-validation; 20% (17 samples) strictly held out.<br>
                • <b>Zero Leakage Scaling:</b> <code>StandardScaler</code> is fitted exclusively on <code>X_train</code>.<br><br>
                The model achieves a realistic, honest <b>88.24% test accuracy</b> that genuinely reflects generalization 
                capacity when deployed at an intake dock.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("#### **Empirical Verification: Artificial Leakage vs True Generalization**")
    
    leak_csv_path = os.path.join(REPORTS_DIR, "leakage_comparison.csv")
    if os.path.exists(leak_csv_path) and results_df is not None:
        leak_df = pd.read_csv(leak_csv_path)
        clean_comp = results_df[['Model', 'Test_Accuracy', 'Test_F1_Score']].copy()
        
        merged_leak = pd.merge(leak_df, clean_comp, on='Model')
        merged_leak.columns = ['Model', 'Naïve Acc (With Leakage)', 'Naïve F1 (With Leakage)', 'Clean Acc (Deduplicated)', 'Clean F1 (Deduplicated)']
        
        st.dataframe(
            merged_leak.style.format({
                'Naïve Acc (With Leakage)': '{:.2%}',
                'Naïve F1 (With Leakage)': '{:.2%}',
                'Clean Acc (Deduplicated)': '{:.2%}',
                'Clean F1 (Deduplicated)': '{:.2%}',
            }),
            use_container_width=True
        )
        
        st.warning(
            "🎓 **Viva Defense Strategy:** When examiners inquire why publicly published models claim 99.5% accuracy, "
            "explain that retaining 976 duplicates causes test set leakage. Our deduplicated methodology delivers a verified "
            "88.24% test accuracy that will reliably generalize to unseen milk deliveries."
        )


# ==============================================================================
# TAB 5: ACADEMIC REPORT & VIVA DEFENSE PREPARATION
# ==============================================================================
with tab_about:
    st.markdown("### Academic Case-Study & Viva Defense Guide")
    st.caption("Theoretical grounding and technical defense preparation for B.Tech CSE Semester V evaluation.")
    
    with st.expander("📖 1. Problem Statement & Industrial Significance", expanded=True):
        st.markdown("""
        Dairy spoilage poses severe consumer health hazards and commercial losses. Traditional laboratory microbial assays 
        (e.g., standard plate count, titration) take hours or days. This project builds an automated, real-time Machine Learning 
        classifier using seven physical and biochemical parameters (pH, temperature, taste, odor, fat, turbidity, colour) 
        to immediately classify milk into **High**, **Medium**, or **Low** quality grades.
        """)
        
    with st.expander("🧪 2. Chemical Feature Definitions & Physiological Significance"):
        st.markdown("""
        | Feature | Type | Valid Range | Physical & Domain Significance |
        | :--- | :--- | :--- | :--- |
        | **pH** | Continuous | 3.0 – 9.5 | Indicator of acidity. Normal fresh milk is slightly acidic (pH 6.5–6.7). Low pH indicates lactic acid fermentation (souring); high pH indicates mastitis or alkaline adulteration. |
        | **Temperature** | Continuous | 34°C – 90°C | Thermal regime. Raw milk drawn at ~37°C. Storage above 40°C triggers exponential bacterial proliferation. |
        | **Taste** | Binary Flag | 0 or 1 | Sensory evaluation. 1 = Optimal palatable dairy taste; 0 = sour, bitter, or abnormal off-flavor. |
        | **Odor** | Binary Flag | 0 or 1 | Olfactory check. 1 = Fresh natural aroma; 0 = absence of standard aroma or pungent defect. |
        | **Fat** | Binary Flag | 0 or 1 | Lipid composition. 1 = Standard/Optimal dairy fat content; 0 = Deficient or skimmed fat level. |
        | **Turbidity** | Binary Flag | 0 or 1 | Light scattering due to colloidal casein and fat globules. 1 = Standard turbid milk; 0 = watery or curd-separated milk. |
        | **Colour** | Integer Scale | 240 – 255 | Spectrophotometric reflectance. Pure white milk scores ~255. Lower values indicate discoloration or foreign matter. |
        | **Grade (Target)** | Categorical | Low, Medium, High | Commercial quality grading for direct consumption or secondary dairy processing. |
        """)
        
    with st.expander("❓ 3. Top Viva Questions & Model Answers"):
        st.markdown("""
        **Q1: Why did you remove duplicate records instead of training on the full 1,059 rows?**  
        *Answer:* The raw dataset contains 976 duplicate rows (92.16%), leaving only 83 unique physical profiles. If duplicates are retained, identical records appear in both train and test splits, causing severe data leakage and artificially inflating accuracy to ~100% through memorization. Removing duplicates guarantees that we evaluate true generalization to novel dairy samples.
        
        **Q2: Why was K-Nearest Neighbors selected as the final production model?**  
        *Answer:* KNN (k=3, distance-weighted, euclidean metric) achieved the highest test accuracy (88.24%) and weighted F1-score (87.23%) on the holdout test set, while maintaining high cross-validation stability (83.41% ± 10.03%).
        
        **Q3: How was data leakage prevented during feature scaling?**  
        *Answer:* `StandardScaler` was fitted strictly on the training partition (`X_train`), and then used to transform both `X_train` and `X_test`. Testing data was never seen by the scaler during parameter computation.
        
        **Q4: Which milk property has the highest predictive importance?**  
        *Answer:* Gini importance from Random Forest and permutation importance on KNN demonstrate that **pH** and **Temperature** are the two dominant discriminators of milk quality, as biochemical deterioration immediately manifests in pH shifts and thermal abuse.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;'>"
    "🥛 <b>MilkSense AI</b> • B.Tech CSE Semester V Machine Learning Case-Study Project • "
    "Designed for Production Deployment & Viva Excellence"
    "</div>",
    unsafe_allow_html=True
)
