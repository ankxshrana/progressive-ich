"""
NeuroSwift.AI — Progressive ICH Detection
Streamlit Application
"""

import io
import base64
import tempfile
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSwift.AI — ICH Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LOGO (embedded) ────────────────────────────────────────────────────────
LOGO_B64 = "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"

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg-deep:      #080c12;
  --bg-mid:       #0d1420;
  --bg-panel:     #111826;
  --bg-card:      #161f2e;
  --border:       #1e2d42;
  --border-glow:  #2a4a6e;
  --accent:       #00b4d8;
  --accent-dim:   #0077a8;
  --red:          #e84040;
  --red-dim:      #a02020;
  --green:        #2dd4a4;
  --amber:        #f5a623;
  --text-primary: #e8edf5;
  --text-secondary: #7a91aa;
  --text-dim:     #3a5068;
  --font-display: 'Syne', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'DM Mono', monospace;
}

html, body, [class*="css"] {
  background-color: var(--bg-deep);
  color: var(--text-primary);
  font-family: var(--font-body);
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-mid); }
::-webkit-scrollbar-thumb { background: var(--border-glow); border-radius: 2px; }

/* Remove default padding */
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Navbar ── */
.ns-navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.8rem 2rem;
  background: linear-gradient(90deg, var(--bg-mid) 0%, #0a1828 100%);
  border-bottom: 1px solid var(--border);
  margin: -2rem -2rem 0 -2rem;
  position: sticky; top: 0; z-index: 999;
  box-shadow: 0 4px 24px #00000066;
}
.ns-navbar-left { display: flex; align-items: center; gap: 1rem; }
.ns-nav-links { display: flex; gap: 1.6rem; }
.ns-nav-link {
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  text-decoration: none;
  padding: 0.25rem 0;
  border-bottom: 1px solid transparent;
  transition: all 0.2s;
}
.ns-nav-link:hover { color: var(--accent); border-bottom-color: var(--accent); }
.ns-badge {
  background: var(--accent-dim);
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.1em;
  padding: 0.2rem 0.6rem;
  border-radius: 2px;
  border: 1px solid var(--accent-dim);
}

/* ── Hero ── */
.ns-hero {
  padding: 2.4rem 0 1.8rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.8rem;
}
.ns-hero-tag {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.22em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.ns-hero h1 {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 800;
  color: var(--text-primary);
  line-height: 1.2;
  margin: 0 0 0.5rem 0;
}
.ns-hero h1 span { color: var(--accent); }
.ns-hero p {
  color: var(--text-secondary);
  font-size: 0.9rem;
  max-width: 600px;
  line-height: 1.6;
}

/* ── Cards ── */
.ns-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s;
}
.ns-card:hover { border-color: var(--border-glow); }
.ns-card-title {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  display: flex; align-items: center; gap: 0.5rem;
}

/* ── Status chips ── */
.ns-chip {
  display: inline-block;
  padding: 0.3rem 0.9rem;
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.ns-chip-red   { background: #2a0808; border: 1px solid var(--red-dim);  color: var(--red);   }
.ns-chip-green { background: #082a1e; border: 1px solid #1a6e56; color: var(--green); }
.ns-chip-blue  { background: #082030; border: 1px solid var(--accent-dim); color: var(--accent); }

/* ── Metric boxes ── */
.ns-metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8rem; margin: 1rem 0; }
.ns-metric {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.9rem 1rem;
}
.ns-metric-label {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  color: var(--text-dim);
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.ns-metric-value {
  font-family: var(--font-display);
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-primary);
}
.ns-metric-unit {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-secondary);
  margin-left: 0.2rem;
}

/* ── Pipeline steps ── */
.ns-pipeline {
  display: flex;
  align-items: center;
  gap: 0;
  overflow-x: auto;
  padding: 0.6rem 0;
  margin: 0.8rem 0;
}
.ns-step {
  display: flex; flex-direction: column; align-items: center;
  gap: 0.3rem; min-width: 90px;
}
.ns-step-icon {
  width: 36px; height: 36px;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
}
.ns-step-label {
  font-family: var(--font-mono);
  font-size: 0.58rem;
  letter-spacing: 0.08em;
  color: var(--text-secondary);
  text-align: center;
  text-transform: uppercase;
}
.ns-arrow {
  font-size: 0.8rem; color: var(--border-glow); padding: 0 0.2rem;
  flex-shrink: 0; margin-bottom: 1.4rem;
}

/* ── Info grid ── */
.ns-info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
.ns-info-item {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.9rem;
}
.ns-info-item h4 {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.4rem 0;
}
.ns-info-item p { color: var(--text-secondary); font-size: 0.82rem; line-height: 1.6; margin: 0; }

/* ── Section headers ── */
.ns-section-header {
  font-family: var(--font-display);
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin: 1.8rem 0 0.8rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 0.6rem;
}
.ns-section-header .ns-dot {
  width: 6px; height: 6px; background: var(--accent); border-radius: 50%;
  flex-shrink: 0;
}

/* ── Upload zone ── */
.stFileUploader > div {
  background: var(--bg-panel) !important;
  border: 1px dashed var(--border-glow) !important;
  border-radius: 6px !important;
}
.stFileUploader label {
  color: var(--text-secondary) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
}

/* ── Buttons ── */
.stButton button {
  background: linear-gradient(135deg, #0a2a40 0%, #0d3a58 100%) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent-dim) !important;
  border-radius: 4px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1.6rem !important;
  transition: all 0.2s !important;
  box-shadow: 0 0 16px #00b4d820 !important;
}
.stButton button:hover {
  background: linear-gradient(135deg, #0d3a58 0%, #104e78 100%) !important;
  box-shadow: 0 0 24px #00b4d840 !important;
  border-color: var(--accent) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg-mid) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Expanders ── */
.streamlit-expanderHeader {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  color: var(--text-secondary) !important;
}

/* ── Disclaimer ── */
.ns-disclaimer {
  background: #1a0808;
  border: 1px solid #4a1818;
  border-left: 3px solid var(--red);
  border-radius: 4px;
  padding: 0.7rem 1rem;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: #cc6666;
  letter-spacing: 0.05em;
  line-height: 1.6;
}

/* ── Scan viewer ── */
.ns-scan-label {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-dim);
  text-align: center;
  padding: 0.3rem 0;
  border-top: 1px solid var(--border);
  margin-top: 0.3rem;
}

/* ── Tables ── */
.ns-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.ns-table th {
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  padding: 0.4rem 0.8rem;
  text-align: left;
}
.ns-table td {
  padding: 0.5rem 0.8rem;
  border-bottom: 1px solid var(--bg-panel);
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 0.78rem;
}
.ns-table tr:hover td { background: var(--bg-panel); color: var(--text-primary); }

/* ── Footer ── */
.ns-footer {
  text-align: center;
  padding: 2rem 0 1rem 0;
  border-top: 1px solid var(--border);
  margin-top: 3rem;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  color: var(--text-dim);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--text-secondary) !important;
  padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── NAVBAR ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ns-navbar">
  <div class="ns-navbar-left">
    <img src="data:image/jpeg;base64,{LOGO_B64}" height="34" alt="NeuroSwift.AI" />
  </div>
  <div class="ns-nav-links">
    <a class="ns-nav-link" href="#dashboard">Dashboard</a>
    <a class="ns-nav-link" href="#clinical-methodology">Methodology</a>
    <a class="ns-nav-link" href="#about">About</a>
  </div>
  <span class="ns-badge">v1.0 RESEARCH</span>
</div>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-hero" id="dashboard">
  <div class="ns-hero-tag">⬡ Intracranial Diagnostics Platform</div>
  <h1>Progressive <span>ICH</span> Detection</h1>
  <p>Upload two sequential Brain CT DICOM scans. The pipeline applies brain windowing, rigid registration, and Siamese U-Net segmentation to quantify hemorrhage progression.</p>
</div>
""", unsafe_allow_html=True)

# Pipeline visual
st.markdown("""
<div class="ns-pipeline">
  <div class="ns-step">
    <div class="ns-step-icon">📂</div>
    <div class="ns-step-label">Load<br>DICOM</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">🪟</div>
    <div class="ns-step-label">Brain<br>Window</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">📐</div>
    <div class="ns-step-label">Rigid<br>Register</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">↔️</div>
    <div class="ns-step-label">Resize<br>512×512</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">🧠</div>
    <div class="ns-step-label">U-Net<br>Segment</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">📊</div>
    <div class="ns-step-label">Area<br>Compare</div>
  </div>
  <div class="ns-arrow">→</div>
  <div class="ns-step">
    <div class="ns-step-icon">🏷️</div>
    <div class="ns-step-label">Classify<br>Result</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── DISCLAIMER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-disclaimer">
  ⚠ RESEARCH USE ONLY — This software is not a certified medical device. Results must not be used as the sole basis for clinical decisions. Always consult a qualified radiologist or neurosurgeon.
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid var(--border);">
      <img src="data:image/jpeg;base64,{LOGO_B64}" width="160" alt="NeuroSwift.AI" />
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.15em;
         text-transform:uppercase; color:#3a5068; padding: 1rem 0 0.4rem 0;">
      ⬡ Configuration
    </div>
    """, unsafe_allow_html=True)

    mode = st.selectbox(
        "Inference Mode",
        ["single", "siamese"],
        help="'single' → 1-channel U-Net per scan. 'siamese' → stack baseline+current as 2 channels for joint inference."
    )

    seg_threshold = st.slider(
        "Segmentation Threshold",
        0.1, 0.9, 0.5, 0.05,
        help="Probability cutoff for positive segmentation pixels."
    )

    prog_threshold = st.slider(
        "Progression Threshold (%)",
        1, 50, 20,
        help="Relative ICH area increase above this % → classified as 'Progressive'."
    )

    weights_file = st.file_uploader(
        "Load .pth Weights (optional)",
        type=["pth", "pt"],
        help="Upload your trained model checkpoint. Without weights, a randomly-initialized network runs in demo mode."
    )

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; letter-spacing:0.15em;
         text-transform:uppercase; color:#3a5068; padding: 1rem 0 0.4rem 0;">
      ⬡ Model Architecture
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <table class="ns-table">
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Type</td><td>MONAI U-Net</td></tr>
      <tr><td>Spatial dims</td><td>2-D</td></tr>
      <tr><td>Channels</td><td>16→32→64→128→256</td></tr>
      <tr><td>Res. units</td><td>2</td></tr>
      <tr><td>Norm</td><td>BatchNorm</td></tr>
      <tr><td>Dropout</td><td>0.1</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
tab_dash, tab_method, tab_about = st.tabs(["📡 Dashboard", "📖 Clinical Methodology", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-card-title">◈ Baseline CT Scan</div>
        </div>
        """, unsafe_allow_html=True)
        baseline_file = st.file_uploader(
            "Upload Baseline DICOM (.dcm)",
            type=["dcm", "dicom"],
            key="baseline",
            label_visibility="collapsed"
        )
        if baseline_file:
            st.markdown('<div style="color:var(--green);font-family:\'DM Mono\',monospace;font-size:0.72rem;padding:0.4rem 0;">✓ Baseline loaded</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="ns-card">
          <div class="ns-card-title">◈ Current CT Scan</div>
        </div>
        """, unsafe_allow_html=True)
        current_file = st.file_uploader(
            "Upload Current DICOM (.dcm)",
            type=["dcm", "dicom"],
            key="current",
            label_visibility="collapsed"
        )
        if current_file:
            st.markdown('<div style="color:var(--green);font-family:\'DM Mono\',monospace;font-size:0.72rem;padding:0.4rem 0;">✓ Current scan loaded</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    run_btn = st.button("⬡ Run ICH Detection Pipeline", use_container_width=True)

    if run_btn:
        if not baseline_file or not current_file:
            st.warning("Please upload both DICOM files before running.")
        else:
            with st.spinner("Running pipeline…"):
                try:
                    import pydicom
                    import SimpleITK as sitk
                    import torch
                    import torch.nn as nn

                    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

                    # ── Save uploads to temp files ──
                    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
                        f.write(baseline_file.read())
                        b_path = f.name
                    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
                        f.write(current_file.read())
                        c_path = f.name

                    # ── PREPROCESSING ──
                    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--accent);">  [1/4] Preprocessing…</div>', unsafe_allow_html=True)

                    def load_dicom(path):
                        ds = pydicom.dcmread(path)
                        px = ds.pixel_array.astype(np.float32)
                        slope = float(getattr(ds, "RescaleSlope", 1.0))
                        interc = float(getattr(ds, "RescaleIntercept", 0.0))
                        try:
                            ps = ds.PixelSpacing
                            spacing = float(ps[0]) * float(ps[1])
                        except AttributeError:
                            spacing = 1.0
                        return px * slope + interc, ds, spacing

                    def brain_window(hu, level=40, width=80):
                        lo, hi = level - width / 2, level + width / 2
                        return ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)

                    def resize_512(img):
                        th, tw = 512, 512
                        h, w = img.shape
                        scale = min(th / h, tw / w)
                        nh, nw = int(round(h * scale)), int(round(w * scale))
                        si = sitk.GetImageFromArray(img)
                        res = sitk.ResampleImageFilter()
                        res.SetSize([nw, nh])
                        res.SetOutputSpacing([si.GetSpacing()[0] * w / nw, si.GetSpacing()[1] * h / nh])
                        res.SetInterpolator(sitk.sitkLinear)
                        res.SetOutputOrigin(si.GetOrigin())
                        res.SetOutputDirection(si.GetDirection())
                        r = sitk.GetArrayFromImage(res.Execute(si))
                        pt, pl = (th - nh) // 2, (tw - nw) // 2
                        canvas = np.zeros((th, tw), dtype=np.float32)
                        canvas[pt:pt + nh, pl:pl + nw] = r
                        return canvas

                    def rigid_register(fixed_arr, moving_arr):
                        fixed = sitk.GetImageFromArray(fixed_arr.astype(np.float32))
                        moving = sitk.GetImageFromArray(moving_arr.astype(np.float32))
                        init = sitk.CenteredTransformInitializer(
                            fixed, moving, sitk.Euler2DTransform(),
                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
                        reg = sitk.ImageRegistrationMethod()
                        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
                        reg.SetMetricSamplingStrategy(reg.RANDOM)
                        reg.SetMetricSamplingPercentage(0.20)
                        reg.SetInterpolator(sitk.sitkLinear)
                        reg.SetShrinkFactorsPerLevel([4, 2, 1])
                        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
                        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
                        reg.SetOptimizerAsGradientDescent(
                            learningRate=1.0, numberOfIterations=200,
                            convergenceMinimumValue=1e-6, convergenceWindowSize=10)
                        reg.SetOptimizerScalesFromPhysicalShift()
                        reg.SetInitialTransform(init, inPlace=False)
                        tx = reg.Execute(
                            sitk.Cast(fixed, sitk.sitkFloat32),
                            sitk.Cast(moving, sitk.sitkFloat32))
                        out = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
                        return sitk.GetArrayFromImage(out).astype(np.float32)

                    bhu, bds, px_spacing = load_dicom(b_path)
                    chu, _, _ = load_dicom(c_path)
                    b512 = resize_512(brain_window(bhu))
                    c_win = brain_window(chu)
                    c512 = rigid_register(b512, resize_512(c_win))

                    # ── MODEL ──
                    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--accent);">  [2/4] Building U-Net…</div>', unsafe_allow_html=True)

                    try:
                        from monai.networks.nets import UNet
                        from monai.networks.layers import Norm
                        in_ch = 1 if mode == "single" else 2
                        net = UNet(
                            spatial_dims=2, in_channels=in_ch, out_channels=1,
                            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
                            num_res_units=2, norm=Norm.BATCH, dropout=0.1)
                        net = nn.Sequential(net, nn.Sigmoid())

                        if weights_file:
                            weights_file.seek(0)
                            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as wf:
                                wf.write(weights_file.read())
                                w_path = wf.name
                            state = torch.load(w_path, map_location=DEVICE)
                            if "model_state_dict" in state:
                                state = state["model_state_dict"]
                            net.load_state_dict(state)
                            weights_status = "✓ Loaded from checkpoint"
                        else:
                            weights_status = "⚠ Random init (demo mode)"

                        net = net.to(DEVICE).eval()

                    except ImportError:
                        st.error("MONAI not installed. Run: pip install monai")
                        st.stop()

                    # ── SEGMENTATION ──
                    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--accent);">  [3/4] Segmenting…</div>', unsafe_allow_html=True)

                    def segment(model, img_b, img_c, m, thr):
                        with torch.no_grad():
                            if m == "siamese":
                                stacked = np.stack([img_b, img_c], axis=0)
                                t = torch.from_numpy(stacked[np.newaxis]).float().to(DEVICE)
                                prob = model(t).squeeze().cpu().numpy()
                                mask_b = (prob >= thr).astype(np.uint8)
                                mask_c = mask_b.copy()
                            else:
                                def _seg(img):
                                    t = torch.from_numpy(img[np.newaxis, np.newaxis]).float().to(DEVICE)
                                    return (model(t).squeeze().cpu().numpy() >= thr).astype(np.uint8)
                                mask_b = _seg(img_b)
                                mask_c = _seg(img_c)
                        return mask_b, mask_c

                    b_mask, c_mask = segment(net, b512, c512, mode, seg_threshold)

                    # ── CLASSIFICATION ──
                    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--accent);">  [4/4] Classifying…</div>', unsafe_allow_html=True)

                    b_px = int(b_mask.sum())
                    c_px = int(c_mask.sum())
                    b_area = b_px * px_spacing
                    c_area = c_px * px_spacing
                    if b_area <= 0:
                        rel_change = float("inf") if c_area > 0 else 0.0
                    else:
                        rel_change = (c_area - b_area) / b_area
                    is_prog = rel_change > (prog_threshold / 100.0)
                    classification = "Progressive ICH" if is_prog else "Stabilized"

                    # Cleanup
                    os.unlink(b_path); os.unlink(c_path)

                    # ══ RENDER RESULTS ══════════════════════════════════════
                    st.markdown('<div class="ns-section-header"><div class="ns-dot"></div>Scan Viewer</div>', unsafe_allow_html=True)

                    red_alpha = ListedColormap([(0, 0, 0, 0), (1, 0.2, 0.2, 0.55)])

                    def render_scan(img, mask, title):
                        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5),
                                               facecolor="#111826")
                        ax.set_facecolor("#111826")
                        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                        ax.imshow(mask, cmap=red_alpha, vmin=0, vmax=1)
                        patch = mpatches.Patch(color=(1, 0.2, 0.2, 0.55), label="ICH")
                        ax.legend(handles=[patch], loc="lower right",
                                  fontsize=7, facecolor="#0d1420",
                                  edgecolor="#1e2d42", labelcolor="#e84040")
                        ax.axis("off")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                                    facecolor="#111826")
                        plt.close(fig)
                        buf.seek(0)
                        return buf

                    vc1, vc2 = st.columns(2, gap="small")
                    with vc1:
                        buf = render_scan(b512, b_mask, "Baseline")
                        st.image(buf, use_container_width=True)
                        st.markdown('<div class="ns-scan-label">Baseline CT — T₀</div>', unsafe_allow_html=True)
                    with vc2:
                        buf = render_scan(c512, c_mask, "Current (Registered)")
                        st.image(buf, use_container_width=True)
                        st.markdown('<div class="ns-scan-label">Current CT — T₁ (Registered)</div>', unsafe_allow_html=True)

                    # Results panel
                    st.markdown('<div class="ns-section-header"><div class="ns-dot"></div>Analysis Results</div>', unsafe_allow_html=True)

                    chip_cls = "ns-chip-red" if is_prog else "ns-chip-green"
                    status_icon = "⚠" if is_prog else "✓"
                    st.markdown(f"""
                    <div style="margin-bottom:1rem;">
                      <span class="ns-chip {chip_cls}">{status_icon} {classification}</span>
                    </div>
                    <div class="ns-metrics">
                      <div class="ns-metric">
                        <div class="ns-metric-label">Baseline Area</div>
                        <div class="ns-metric-value">{b_area:.1f}<span class="ns-metric-unit">mm²</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a5068;">{b_px:,} px</div>
                      </div>
                      <div class="ns-metric">
                        <div class="ns-metric-label">Current Area</div>
                        <div class="ns-metric-value">{c_area:.1f}<span class="ns-metric-unit">mm²</span></div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a5068;">{c_px:,} px</div>
                      </div>
                      <div class="ns-metric">
                        <div class="ns-metric-label">Δ Relative Change</div>
                        <div class="ns-metric-value" style="color:{'var(--red)' if is_prog else 'var(--green)'}">
                          {rel_change*100:+.1f}<span class="ns-metric-unit">%</span>
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a5068;">
                          Δabs: {c_area-b_area:+.1f} mm²
                        </div>
                      </div>
                    </div>
                    <div class="ns-card" style="margin-top:0.8rem;">
                      <div class="ns-card-title">◈ Pipeline Log</div>
                      <table class="ns-table">
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Inference mode</td><td>{mode}</td></tr>
                        <tr><td>Segmentation threshold</td><td>{seg_threshold}</td></tr>
                        <tr><td>Progression threshold</td><td>{prog_threshold}%</td></tr>
                        <tr><td>Pixel spacing</td><td>{px_spacing:.4f} mm²/px</td></tr>
                        <tr><td>Compute device</td><td>{DEVICE.upper()}</td></tr>
                        <tr><td>Weights</td><td>{weights_status}</td></tr>
                      </table>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    import traceback
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    else:
        # Placeholder dual viewer
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem;">
          <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:6px;
               height:320px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:0.6rem;">
            <div style="font-size:2rem;opacity:0.2;">🧠</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                 text-transform:uppercase;color:var(--text-dim);">Awaiting Baseline</div>
          </div>
          <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:6px;
               height:320px;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:0.6rem;">
            <div style="font-size:2rem;opacity:0.2;">🧠</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                 text-transform:uppercase;color:var(--text-dim);">Awaiting Current Scan</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLINICAL METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown("""
    <div class="ns-hero" id="clinical-methodology">
      <div class="ns-hero-tag">⬡ For Clinical Reviewers</div>
      <h1>Clinical <span>Methodology</span></h1>
      <p>A transparent explanation of each stage in the automated analysis pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ns-section-header"><div class="ns-dot"></div>Stage 1 — Image Normalization (HU → Grayscale)</div>
    <div class="ns-info-grid">
      <div class="ns-info-item">
        <h4>Hounsfield Unit Conversion</h4>
        <p>Raw DICOM pixel values are converted to Hounsfield Units (HU) using the linear rescale parameters stored in the DICOM header: <strong>HU = pixel × RescaleSlope + RescaleIntercept</strong>.</p>
      </div>
      <div class="ns-info-item">
        <h4>Brain Windowing</h4>
        <p>A window of <strong>Level 40 HU, Width 80 HU</strong> (range −0 to +80 HU) is applied. This isolates the density range relevant to blood vs. brain parenchyma, suppressing bone and air artefacts.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\text{Normalized} = \frac{\text{HU} - (\text{Level} - \frac{\text{Width}}{2})}{\text{Width}}")

    st.markdown("""
    <div class="ns-section-header"><div class="ns-dot"></div>Stage 2 — Spatial Alignment (Rigid Registration)</div>
    <div class="ns-card">
      <p style="color:var(--text-secondary);font-size:0.85rem;line-height:1.7;margin:0;">
        The current scan is rigidly registered to the baseline using <strong>SimpleITK's Euler2D transform</strong>
        initialized with geometry-based centering. Optimization uses <strong>Mattes Mutual Information</strong>
        as the metric (64 histogram bins, 20% random sampling) with a 3-level multi-resolution pyramid
        (shrink factors 4/2/1, Gaussian smoothing σ = 2/1/0 mm). This corrects for patient
        repositioning between scans while preserving internal anatomy, ensuring that any detected
        area change is due to hemorrhage growth rather than scan geometry differences.
      </p>
    </div>

    <div class="ns-section-header"><div class="ns-dot"></div>Stage 3 — Siamese U-Net Segmentation</div>
    <div class="ns-info-grid">
      <div class="ns-info-item">
        <h4>Architecture</h4>
        <p>A 2-D <strong>MONAI U-Net</strong> with encoder channels (16→32→64→128→256), stride-2 downsampling, 2 residual units per block, Batch Normalization, and dropout = 0.1. Output activates through Sigmoid for probabilistic segmentation.</p>
      </div>
      <div class="ns-info-item">
        <h4>Siamese Mode</h4>
        <p>In siamese mode, the baseline and current slices are <strong>channel-stacked</strong> into a single 2-channel tensor (in_channels=2). The network learns joint features across both timepoints, improving detection of subtle interval changes.</p>
      </div>
    </div>

    <div class="ns-section-header"><div class="ns-dot"></div>Stage 4 — Progression Quantification</div>
    """, unsafe_allow_html=True)

    st.latex(r"\text{Progression \%} = \left( \frac{A_{\text{current}} - A_{\text{baseline}}}{A_{\text{baseline}}} \right) \times 100")

    st.markdown("""
    <div class="ns-card" style="margin-top:0.8rem;">
      <div class="ns-card-title">◈ Classification Logic</div>
      <table class="ns-table">
        <tr><th>Condition</th><th>Classification</th><th>Clinical Interpretation</th></tr>
        <tr>
          <td>Progression % &gt; threshold (default 20%)</td>
          <td><span class="ns-chip ns-chip-red">Progressive ICH</span></td>
          <td>Hemorrhage has significantly expanded. Urgent neurosurgical review recommended.</td>
        </tr>
        <tr>
          <td>Progression % ≤ threshold</td>
          <td><span class="ns-chip ns-chip-green">Stabilized</span></td>
          <td>No significant interval change detected. Continue observation protocol.</td>
        </tr>
        <tr>
          <td>Baseline area = 0, current &gt; 0</td>
          <td><span class="ns-chip ns-chip-red">Progressive ICH</span></td>
          <td>New hemorrhage detected (no prior baseline lesion).</td>
        </tr>
      </table>
    </div>

    <div class="ns-section-header"><div class="ns-dot"></div>Overlay Colormap</div>
    <div class="ns-card">
      <p style="color:var(--text-secondary);font-size:0.85rem;line-height:1.7;margin:0 0 0.6rem 0;">
        Segmentation masks are displayed using a semi-transparent red colormap to maximize contrast
        against the grayscale brain CT without obscuring underlying anatomy.
      </p>
      <div style="display:flex;align-items:center;gap:1rem;">
        <div style="width:60px;height:20px;background:rgba(255,51,51,0.55);border-radius:2px;border:1px solid #a02020;"></div>
        <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-secondary);">
          RGBA = (1.0, 0.2, 0.2, 0.55) — ICH Segmentation Mask
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"""
    <div class="ns-hero" id="about">
      <div class="ns-hero-tag">⬡ Platform Information</div>
      <h1>About <span>NeuroSwift.AI</span></h1>
    </div>
    <div style="text-align:center;padding:1.5rem 0 2rem 0;">
      <img src="data:image/jpeg;base64,{LOGO_B64}" width="220" alt="NeuroSwift.AI" />
      <div style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.2em;
           text-transform:uppercase;color:var(--text-dim);margin-top:0.8rem;">
        The Intelligent Edge in Intracranial Diagnostics
      </div>
    </div>

    <div class="ns-info-grid">
      <div class="ns-info-item">
        <h4>Platform</h4>
        <p>NeuroSwift.AI is a research-grade AI platform for progressive intracranial hemorrhage detection using serial Brain CT DICOM imaging.</p>
      </div>
      <div class="ns-info-item">
        <h4>Intended Use</h4>
        <p>Designed as a decision-support tool for radiologists and neurosurgeons reviewing serial CT scans. Not a standalone diagnostic device.</p>
      </div>
      <div class="ns-info-item">
        <h4>Technology</h4>
        <p>Built on PyTorch + MONAI, SimpleITK registration, pydicom DICOM parsing, and Streamlit for rapid clinical prototyping.</p>
      </div>
      <div class="ns-info-item">
        <h4>Training Data</h4>
        <p>U-Net architecture compatible with PhysioNet ICH dataset and RSNA Intracranial Hemorrhage Detection challenge datasets.</p>
      </div>
    </div>

    <div class="ns-section-header" style="margin-top:1.5rem;"><div class="ns-dot"></div>Loading Custom Weights (.pth)</div>
    <div class="ns-card">
      <div class="ns-card-title">◈ How to Load Your Trained Model</div>
      <p style="color:var(--text-secondary);font-size:0.85rem;line-height:1.8;margin:0 0 0.8rem 0;">
        Upload your <code>.pth</code> file via the sidebar. The app expects a state dict in one of two formats:
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""# Format 1 — Raw state dict
torch.save(model.state_dict(), 'weights.pth')

# Format 2 — Checkpoint with metadata (also supported)
torch.save({
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.12,
}, 'checkpoint.pth')
""", language="python")

    st.markdown("""
    <div class="ns-card" style="margin-top:0.6rem;">
      <div class="ns-card-title">◈ Input Shape Requirements</div>
      <table class="ns-table">
        <tr><th>Mode</th><th>Expected Tensor Shape</th><th>Description</th></tr>
        <tr><td>single</td><td>(B, 1, 512, 512)</td><td>One channel per scan; two independent forward passes</td></tr>
        <tr><td>siamese</td><td>(B, 2, 512, 512)</td><td>Baseline + Current stacked as channel dim; one forward pass</td></tr>
      </table>
    </div>

    <div class="ns-section-header" style="margin-top:1.5rem;"><div class="ns-dot"></div>Recommended Training Datasets</div>
    <table class="ns-table">
      <tr><th>Dataset</th><th>Source</th><th>Notes</th></tr>
      <tr><td>RSNA ICH Challenge</td><td>Kaggle / RSNA</td><td>Multi-type ICH labels, 25,000+ studies</td></tr>
      <tr><td>PhysioNet ICH</td><td>physionet.org</td><td>Sequential imaging with progression labels</td></tr>
      <tr><td>CQ500</td><td>qure.ai</td><td>Diverse real-world emergency CT scans</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-footer">
  NEUROSWIFT.AI &nbsp;·&nbsp; RESEARCH PROTOTYPE &nbsp;·&nbsp; NOT FOR CLINICAL USE &nbsp;·&nbsp;
  PIPELINE: PYDICOM + SIMPLEITK + MONAI + TORCH
</div>
""", unsafe_allow_html=True)
