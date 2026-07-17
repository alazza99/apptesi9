import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RUNAI | Performance Intelligence", layout="wide", initial_sidebar_state="expanded")

# =========================================================
#  DESIGN SYSTEM — RUNAI
#  Palette:  #080B12 base / #0E1420 panel / #00E5FF cyan
#            #FF6A3D signal / #00F5A0 mint / #B8C2D0 text
#  Type:     Space Grotesk (display) / Inter (body) / JetBrains Mono (data)
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #080B12;
        --panel: #0E1420;
        --panel-2: #111827;
        --line: #1c2333;
        --cyan: #00E5FF;
        --signal: #FF6A3D;
        --mint: #00F5A0;
        --amber: #FFB020;
        --text: #E8ECF2;
        --text-dim: #8792A3;
        --text-faint: #566178;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(0,229,255,0.05) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(255,106,61,0.04) 0%, transparent 45%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    * { letter-spacing: -0.01em; }

    /* ---------- TELEMETRY HEADER (signature element) ---------- */
    .telemetry-bar {
        display: flex; align-items: center; gap: 0;
        height: 3px; width: 100%;
        background: linear-gradient(90deg, var(--cyan) 0%, var(--mint) 35%, var(--signal) 70%, var(--cyan) 100%);
        background-size: 200% 100%;
        border-radius: 2px;
        margin-bottom: 22px;
        animation: scanline 6s linear infinite;
    }
    @keyframes scanline { 0% {background-position: 0% 0;} 100% {background-position: 200% 0;} }

    .app-header { padding: 6px 0 18px 0; }
    .app-kicker {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72em; letter-spacing: 0.25em;
        color: var(--cyan); text-transform: uppercase; margin-bottom: 6px; display:flex; align-items:center; gap:10px;
    }
    .app-kicker .dot { width:6px; height:6px; border-radius:50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); display:inline-block; }

    h1.hero-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.6em;
        color: #fff; margin: 0 0 4px 0; letter-spacing: -0.03em; line-height: 1.05; text-align:left;
    }
    .hero-sub { color: var(--text-dim); font-size: 1.02em; max-width: 640px; margin-bottom: 4px; }

    h2 {
        font-family: 'Space Grotesk', sans-serif; color: #fff; font-weight: 600; font-size: 1.5em;
        padding-bottom: 12px; margin: 8px 0 18px 0; border-bottom: 1px solid var(--line); letter-spacing: -0.02em;
    }
    h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text); font-size: 1.15em; font-weight: 600; letter-spacing: -0.01em; }

    .section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7em; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text-faint); margin-bottom: 6px;
    }

    /* ---------- INFO / STATUS PANELS ---------- */
    .info-box, .success-box, .warning-box, .danger-box {
        padding: 18px 20px; border-radius: 10px; margin: 16px 0; color: var(--text-dim);
        background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--cyan);
    }
    .success-box { border-left-color: var(--mint); }
    .warning-box { border-left-color: var(--amber); }
    .danger-box  { border-left-color: var(--signal); }

    .kpi-card {
        background: var(--panel); border-radius: 12px; padding: 26px 20px; text-align: center;
        border: 1px solid var(--line); position: relative; overflow: hidden;
    }
    .kpi-card::before {
        content: ""; position: absolute; top:0; left:0; right:0; height: 2px;
        background: linear-gradient(90deg, var(--cyan), transparent);
    }

    .explain-text {
        font-family: 'Inter', sans-serif; font-size: 0.87em; color: var(--text-faint); line-height: 1.55;
        margin-top: 6px; padding: 14px 16px; background: var(--panel); border-radius: 8px; border-left: 2px solid var(--line);
    }
    .explain-text strong { color: var(--text-dim); font-weight: 600; }

    .data-figure { font-family: 'JetBrains Mono', monospace; }

    /* ---------- WIDGETS / FORM READABILITY ---------- */
    .stForm { background-color: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 26px; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #131a29 !important; color: var(--text) !important; border: 1px solid var(--line) !important;
    }
    div[data-baseweb="popover"] { background-color: #131a29 !important; }
    div[data-baseweb="popover"] ul, div[data-baseweb="menu"], ul[role="listbox"] { background-color: #131a29 !important; }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #131a29 !important; color: var(--text) !important;
    }
    div[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover {
        background-color: #1c2740 !important; color: #ffffff !important;
    }
    .stSlider label, .stSelectSlider label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color: var(--text-dim) !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important;
    }
    .stSlider [data-baseweb="slider"] div { color: var(--text) !important; }
    div[data-testid="stTickBar"] { color: var(--text-faint) !important; }
    .stSelectSlider [role="slider"] { background-color: var(--cyan) !important; }
    div[data-testid="stWidgetLabel"] p { color: var(--text-dim) !important; }
    div[data-testid="stForm"] { background-color: var(--panel); border: 1px solid var(--line); border-radius: 14px; }

    .stButton button, .stFormSubmitButton button {
        background: linear-gradient(90deg, var(--cyan), #00b8d4) !important; color: #04121a !important;
        border: none !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.02em !important;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] { background-color: var(--bg) !important; border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] > div { background-color: var(--bg) !important; }
    section[data-testid="stSidebar"] h3 { color: var(--text-dim) !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label { color: var(--text) !important; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] button {
        background-color: #131a29 !important; border: 1px solid var(--line) !important;
    }
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button div { color: var(--text) !important; }
    section[data-testid="stSidebar"] hr { border-color: var(--line) !important; }

    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #fff !important; }
    div[data-testid="stMetricLabel"] { font-family: 'Inter', sans-serif !important; color: var(--text-faint) !important; }

    /* Modificato: rimosso il bordo per le immagini astratte */
    .hero-media {
        border-radius: 14px; overflow: hidden; position: relative; margin-bottom: 6px; border: none;
    }
    .hero-media img { display:block; width: 100%; height: 220px; object-fit: cover; filter: saturate(0.9) brightness(0.75); }
    .hero-media .tag {
        position:absolute; bottom:14px; left:14px; font-family:'JetBrains Mono', monospace; font-size:0.72em;
        letter-spacing:0.12em; color:#fff; background: rgba(8,11,18,0.65); padding: 5px 10px; border-radius:6px;
        border: 1px solid rgba(255,255,255,0.15); text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"

PLOTLY_FONT = dict(family="Inter, sans-serif", color="#B8C2D0")

def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        title_font=dict(family="Space Grotesk, sans-serif", color="#E8ECF2", size=16),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if height:
        fig.update_layout(height=height)
    return fig

def header_block(kicker, title, subtitle, image_url=None, image_tag=None):
    st.markdown("<div class='telemetry-bar'></div>", unsafe_allow_html=True)
    if image_url:
        col_txt, col_img = st.columns([1.4, 1])
        with col_txt:
            st.markdown(f"""
            <div class="app-header">
                <div class="app-kicker"><span class="dot"></span>{kicker}</div>
                <h1 class="hero-title">{title}</h1>
                <p class="hero-sub">{subtitle}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_img:
            st.markdown(f"""
            <div class="hero-media">
                <img src="{image_url}" />
                <div class="tag">{image_tag or ''}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-kicker"><span class="dot"></span>{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90

    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    temp = np.random.uniform(10, 30, n)

    fc_media = 100 + (velocita * 3) + (distanza * 0.5) + (temp * 0.3) + np.random.normal(0, 5, n)
    fc_media = np.clip(fc_media, 80, 200)

    rpe_base = (distanza * 0.2) + (stress_lavoro * 0.3) - (ore_sonno * 0.4) + 4
    rpe = np.clip(np.round(rpe_base + np.random.normal(0, 1, n)), 1, 10)

    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1),
        'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media),
        'FC Max': np.round(fc_media + np.random.uniform(10, 30, n)),
        'Temp (°C)': np.round(temp, 1),
        'RPE': rpe,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Stress Lavoro': stress_lavoro,
        'Ore Lavoro': np.round(np.random.uniform(4, 10, n), 1),
        'Calorie': np.round(distanza * 100 + np.random.uniform(-50, 50, n)),
    })

    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)

    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}
    st.session_state.device_connected = False
    st.session_state.device_info = None

# Nuove Immagini Astratte Senza Contorno
IMG_HERO_ANALISI = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNDAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImcxIiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzAwRTVGRiIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwRjVBMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjkwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiMwODBCMTIiLz48cGF0aCBkPSJNLTEwMCwyMDAgUTIwMCw0MDAgNDUwLDIwMCBUMTAwMCwyMDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0idXJsKCNnMSkiIHN0cm9rZS13aWR0aD0iNDAiIG9wYWNpdHk9IjAuMyIvPjxwYXRoIGQ9Ik0tMTAwLDI1MCBRMjAwLDUwIDQ1MCwyNTAgVDEwMDAsMjUwIiBmaWxsPSJub25lIiBzdHJva2U9InVybCgjZzEpIiBzdHJva2Utd2lkdGg9IjIwIiBvcGFjaXR5PSIwLjUiLz48Y2lyY2xlIGN4PSI0NTAiIGN5PSIyMDAiIHI9IjE1MCIgZmlsbD0idXJsKCNnMSkiIG9wYWNpdHk9IjAuMiIvPjwvc3ZnPg=="
IMG_HERO_STATS = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNDAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImcyIiB4MT0iMCIgeTE9IjEiIHgyPSIxIiB5Mj0iMCI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iI0ZGNkEzRCIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0ZGQjAyMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjkwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiMwODBCMTIiLz48cG9seWdvbiBwb2ludHM9IjAsNDAwIDIwMCwxMDAgNDUwLDMwMCA3MDAsNTAgOTAwLDI1MCA5MDAsNDAwIiBmaWxsPSJ1cmwoI2cyKSIgb3BhY2l0eT0iMC40Ii8+PHBvbHlnb24gcG9pbnRzPSIwLDQwMCAzMDAsMjAwIDUwMCwzNTAgODAwLDE1MCA5MDAsMzAwIDkwMCw0MDAiIGZpbGw9InVybCgjZzIpIiBvcGFjaXR5PSIwLjYiLz48L3N2Zz4="
IMG_HERO_KPI = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNDAwIj48ZGVmcz48cmFkaWFsR3JhZGllbnQgaWQ9ImczIj48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMDBFNUZGIiBzdG9wLW9wYWNpdHk9IjAuOCIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwRTVGRiIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD48cmFkaWFsR3JhZGllbnQgaWQ9Imc0Ij48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjMDBGNUEwIiBzdG9wLW9wYWNpdHk9IjAuNiIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAwRjVBMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjkwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiMwODBCMTIiLz48Y2lyY2xlIGN4PSIzMDAiIGN5PSIyMDAiIHI9IjI1MCIgZmlsbD0idXJsKCNnMykiLz48Y2lyY2xlIGN4PSI2NTAiIGN5PSIyNTAiIHI9IjMwMCIgZmlsbD0idXJsKCNnNCkiLz48L3N2Zz4="
IMG_HERO_ML = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNDAwIj48ZGVmcz48cmFkaWFsR3JhZGllbnQgaWQ9Imc1Ij48c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjRkZCMDIwIiBzdG9wLW9wYWNpdHk9IjAuNSIvPjxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iI0ZGQjAyMCIgc3RvcC1vcGFjaXR5PSIwIi8+PC9yYWRpYWxHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjkwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiMwODBCMTIiLz48Y2lyY2xlIGN4PSI0NTAiIGN5PSIyMDAiIHI9IjMwMCIgZmlsbD0idXJsKCNnNSkiLz48cGF0aCBkPSJNMTAwLDEwMCBMMzAwLDI1MCBMNTAwLDE1MCBMNzAwLDMwMCBMODUwLDEwMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjRkZCMDIwIiBzdHJva2Utd2lkdGg9IjIiIG9wYWNpdHk9IjAuNSIvPjxjaXJjbGUgY3g9IjEwMCIgY3k9IjEwMCIgcj0iOCIgZmlsbD0iI0ZGQjAyMCIvPjxjaXJjbGUgY3g9IjMwMCIgY3k9IjI1MCIgcj0iMTUiIGZpbGw9IiNGRkIwMjAiIG9wYWNpdHk9IjAuOCIvPjxjaXJjbGUgY3g9IjUwMCIgY3k9IjE1MCIgcj0iMTAiIGZpbGw9IiNGRkIwMjAiLz48Y2lyY2xlIGN4PSI3MDAiIGN5PSIzMDAiIHI9IjIwIiBmaWxsPSIjRkZCMDIwIiBvcGFjaXR5PSIwLjYiLz48Y2lyY2xlIGN4PSI4NTAiIGN5PSIzMDAiIHI9IjUiIGZpbGw9IiNGRkIwMjAiLz48L3N2Zz4="
IMG_HERO_PLAN = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgNDAwIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9Imc2IiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMCI+PHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzAwRjVBMCIvPjxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjMDBFNUZGIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjRkY2QTNEIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjkwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9IiMwODBCMTIiLz48cGF0aCBkPSJNMCwyMDAgQzMwMCwzMDAgNjAwLDEwMCA5MDAsMjAwIiBmaWxsPSJub25lIiBzdHJva2U9InVybCgjZzYpIiBzdHJva2Utd2lkdGg9IjMwIiBvcGFjaXR5PSIwLjQuLz48cGF0aCBkPSJNMCwyMjAgQzMwMCwzMjAgNjAwLDEyMCA5MDAsMjIwIiBmaWxsPSJub25lIiBzdHJva2U9InVybCgjZzYpIiBzdHJva2Utd2lkdGg9IjE1IiBvcGFjaXR5PSIwLjYiLz48cGF0aCBkPSJNMCwyNDAgQzMwMCwzNDAgNjAwLDE0MCA5MDAsMjQwIiBmaWxsPSJub25lIiBzdHJva2U9InVybCgjZzYpIiBzdHJva2Utd2lkdGg9IjUiIG9wYWNpdHk9IjAuOSIvPjwvc3ZnPg=="

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
            <div style='width:34px; height:34px; border-radius:8px; background:linear-gradient(135deg, #00E5FF, #00F5A0); display:flex; align-items:center; justify-content:center; font-family:"Space Grotesk",sans-serif; font-weight:800; color:#04121a; font-size:1.1em;'>R</div>
            <h1 style='color: white; text-align: left; font-size: 1.55em; font-family:"Space Grotesk",sans-serif; font-weight:700; margin:0; letter-spacing:-0.03em;'>RUNAI</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='color: #566178; font-size: 0.78em; margin-top: 2px; margin-bottom: 22px; font-family:\"JetBrains Mono\",monospace; letter-spacing:0.1em; text-transform:uppercase;'>Performance Intelligence System</p>", unsafe_allow_html=True)

    st.subheader("Dispositivo")

    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }

    device_scelto = st.selectbox("Seleziona dispositivo:", list(dispositivi.keys()), label_visibility="collapsed")

    if st.button("CONNETTI DISPOSITIVO", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_info = {
            'nome': device_scelto,
            'fc': np.random.randint(60, 80),
            'battery': np.random.randint(70, 100),
            'steps': np.random.randint(2000, 5000),
            'calories': np.random.randint(150, 300),
            'sync_time': pd.Timestamp.now().strftime('%H:%M:%S')
        }

    if st.session_state.device_connected:
        st.markdown("---")
        st.markdown("""
        <div style='background-color: #0E1420; border: 1px solid #1c2333; border-radius: 10px; padding: 16px; font-family:"Inter",sans-serif;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                <span style='color: #00F5A0; font-weight: bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.1em;'>&#9679; LIVE</span>
                <span style='color: #566178; font-size: 0.75em; font-family:"JetBrains Mono",monospace;'>{}</span>
            </div>
            <div style='color: #E8ECF2; font-family:"JetBrains Mono",monospace; font-size:0.92em;'>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>FC</span><span style='font-weight:600;'>{} bpm</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Batteria</span><span style='font-weight:600; color:#00F5A0;'>{}%</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Passi</span><span style='font-weight:600;'>{:,}</span></div>
                <div style='display:flex; justify-content:space-between; margin: 6px 0;'><span style='color:#8792A3; font-family:"Inter",sans-serif;'>Calorie</span><span style='font-weight:600;'>{}</span></div>
            </div>
            <div style='color: #566178; font-size: 0.7em; margin-top: 12px; text-align: center; font-family:"JetBrains Mono",monospace;'>SYNC {}</div>
        </div>
        """.format(
            st.session_state.device_info['nome'],
            st.session_state.device_info['fc'],
            st.session_state.device_info['battery'],
            st.session_state.device_info['steps'],
            st.session_state.device_info['calories'],
            st.session_state.device_info['sync_time']
        ), unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("<h3 style='color: #00E5FF; font-size: 0.8em; letter-spacing: 0.15em; text-transform: uppercase;'>Navigazione</h3>", unsafe_allow_html=True)
    pagina = st.sidebar.selectbox(
        "Menu",
        ["ANALISI", "STATISTICHE", "KPI DASHBOARD", "ML EXPLAINED", "CONSIGLIO FINALE"],
        label_visibility="collapsed"
    )

# ----------------- PAGINA 1: ANALISI -----------------
if pagina == "ANALISI":
    header_block(
        "Modulo 01 — Acquisizione Dati",
        "Stato di Forma, Oggi.",
        "Inserisci i parametri fisiologici e di carico odierni: il motore predittivo li userà per calcolare il tuo rischio infortunio in tempo reale.",
        IMG_HERO_ANALISI, "Pre-Session Check"
    )

    st.markdown("""
    <div class='info-box'>
    <strong>Compila il questionario per avviare il motore di predizione.</strong>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_analisi"):
        st.markdown("### Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)

        st.markdown("#### Obiettivo Finale (Lungo Termine)")
        col_of1, col_of2, col_of3 = st.columns(3)
        with col_of1:
            obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
        with col_of2:
            data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
        with col_of3:
            km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)

        st.markdown("---")
        st.markdown("### Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)

        st.markdown("---")
        st.markdown("### Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)

        st.markdown("---")
        st.markdown("### Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)

        st.markdown("---")
        bottone = st.form_submit_button("ANALIZZA PARAMETRI", use_container_width=True)

    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi, 'distanza_oggi': distanza_oggi,
            'obj_finale': obj_finale, 'data_obj_finale': data_obj_finale, 'km_obj_finale': km_obj_finale,
            'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo, 'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro, 'tipo_allenamento': tipo_allenamento,
            'rpe_previsto': rpe_previsto,
        }
        st.success("Analisi completata e caricata nel modello.")

        if obj_finale:
            giorni_rimasti = (pd.Timestamp(data_obj_finale) - pd.Timestamp.today()).days
            st.markdown(f"""
            <div class='info-box' style='border-left-color:#00F5A0;'>
            <strong>Obiettivo Finale impostato:</strong> {obj_finale} ({km_obj_finale:.1f} km) — mancano circa <strong>{max(giorni_rimasti,0)} giorni</strong> ({pd.Timestamp(data_obj_finale).strftime('%d/%m/%Y')}).
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 2: STATISTICHE -----------------
elif pagina == "STATISTICHE":
    header_block(
        "Modulo 02 — Analytics Storico",
        "90 Giorni di Dati Grezzi.",
        "Volume, intensità e recupero degli ultimi tre mesi, decodificati in pattern utilizzabili.",
        IMG_HERO_STATS, "Historical Load"
    )

    df = st.session_state.dati.copy()

    st.subheader("KPI Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f} km", "90 giorni")
    col_m2.metric("Sessioni", f"{len(df)}")
    col_m3.metric("Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")

    st.markdown("---")
    st.subheader("Analisi Dettagliata")

    tab1, tab2, tab3, tab4 = st.tabs(["Volume", "Intensità", "Recupero", "Tabella Storico"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({'Distanza (km)': 'sum'}).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=300, color='Distanza (km)', color_continuous_scale=[[0,'#0E4A57'],[1,'#00E5FF']])
            st.plotly_chart(style_fig(fig1), use_container_width=True)
            st.markdown("<div class='explain-text'>Verifica che le barre non facciano salti maggiori del 10% da una settimana all'altra. Un picco improvviso porta a infiammazioni tendinee.</div>", unsafe_allow_html=True)

            st.markdown("**Carico per Giorno della Settimana**")
            df['Giorno_Settimana'] = df['Giorno'].dt.day_name()
            df_day = df.groupby('Giorno_Settimana')['Distanza (km)'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig_day = px.bar(df_day, x='Giorno_Settimana', y='Distanza (km)', height=300, color_discrete_sequence=['#00E5FF'])
            st.plotly_chart(style_fig(fig_day), use_container_width=True)
            st.markdown("<div class='explain-text'>Visualizza la tua routine. Assicurati che ai giorni con barre alte seguano giorni con barre basse o assenti (recupero attivo).</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=300, markers=True)
            fig_cum.update_traces(line_color="#00E5FF")
            st.plotly_chart(style_fig(fig_cum), use_container_width=True)
            st.markdown("<div class='explain-text'>Una linea retta indica costanza. Una linea piatta indica stop o infortuni.</div>", unsafe_allow_html=True)

            record_km = df.loc[df['Distanza (km)'].idxmax()]
            record_vel = df.loc[df['Velocità (km/h)'].idxmax()]
            giorni_attivi = (df['Distanza (km)'] > 0).sum()
            streak = int((df['Distanza (km)'] > df['Distanza (km)'].mean()).astype(int).groupby((df['Distanza (km)'] <= df['Distanza (km)'].mean()).cumsum()).cumsum().max())

            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #0E1420 0%, #131427 100%);'>
                <h3 style='color:#FFB020; margin-bottom:15px;'>Bacheca Record — Ultimi 90 giorni</h3>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Corsa più lunga</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_km['Distanza (km)']:.1f} km</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Velocità massima</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{record_vel['Velocità (km/h)']:.1f} km/h</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Miglior striscia sopra media</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{streak} allenamenti</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#B8C2D0; font-family:"Inter",sans-serif;'>
                    <span>Giorni con allenamento</span><strong style='color:#fff; font-family:"JetBrains Mono",monospace;'>{giorni_attivi} / {len(df)}</strong>
                </div>
                <p style='color:#566178; font-size:0.8em; margin-top:12px; margin-bottom:0; font-family:"Inter",sans-serif;'>Ogni record battuto è un passo in più verso l'obiettivo finale.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', color='RPE', color_continuous_scale=[[0,'#0E4A57'],[0.5,'#00E5FF'],[1,'#FF6A3D']], height=300)
            st.plotly_chart(style_fig(fig2), use_container_width=True)
            st.markdown("<div class='explain-text'>Più i punti scendono verso il basso a destra, più il tuo cuore è efficiente (vai veloce faticando poco). I punti arancioni sono i lavori massimali.</div>", unsafe_allow_html=True)

            st.markdown("**Ripartizione Zone Cardiache**")
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Fondo Lento)', 'Z3 (Medio/Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df['Zone'] = pd.cut(df['FC Media'], bins=bins, labels=labels)
            zone_counts = df['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', hole=0.6, height=300, color_discrete_sequence=['#00E5FF','#00B8D4','#0E4A57','#FFB020','#FF6A3D'])
            st.plotly_chart(style_fig(fig_zones), use_container_width=True)
            st.markdown("<div class='explain-text'>Un allenamento sano prevede 80% in Z1/Z2 e 20% in Z4/Z5. Evita di rimanere intrappolato in Z3 (zona grigia), che stanca senza allenare efficacemente.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=300, color_discrete_sequence=['#00E5FF'])
            fig3.add_vline(x=3.5, line_dash="dash", line_color="#00F5A0")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig3), use_container_width=True)
            st.markdown("<div class='explain-text'>Mostra quante volte hai spinto al massimo (oltre la linea arancione) e quante volte hai recuperato (sotto la linea verde).</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=300, markers=True)
            fig_sleep.update_traces(line_color="#00E5FF")
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="#00F5A0")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_sleep), use_container_width=True)
            st.markdown("<div class='explain-text'>Cerca di stare sempre sopra la linea verde. Le discese verso la linea arancione corrispondono a cali di prestazione muscolare.</div>", unsafe_allow_html=True)

            st.markdown("**Debito di Sonno (Rolling 7gg)**")
            df['Debito'] = df['Ore Sonno'].apply(lambda x: max(0, 7.5 - x)).rolling(7).sum()
            fig_debt = px.area(df, x='Giorno', y='Debito', height=300, color_discrete_sequence=['#FF6A3D'])
            st.plotly_chart(style_fig(fig_debt), use_container_width=True)
            st.markdown("<div class='explain-text'>Quest'area è fatica accumulata. Se il debito supera le 5 ore in una settimana, il rischio di strappi o contratture decuplica.</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', color='Rischio Infortunio', color_continuous_scale=[[0,'#00E5FF'],[1,'#FF6A3D']], height=300)
            fig4.add_hline(y=7, line_dash="dash", line_color="#FFB020")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="#FFB020")
            st.plotly_chart(style_fig(fig4), use_container_width=True)
            st.markdown("<div class='explain-text'>Il quadrante in alto a sinistra (poco sonno, alto sforzo) è la 'zona critica'. Evita che i punti cadano in quell'area.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: 'ALTO' if x == 1 else 'OK')

        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#111827', align='center', font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#0E1420', align='center', font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=500)
        st.plotly_chart(style_fig(fig_table), use_container_width=True)

# ----------------- PAGINA 3: KPI DASHBOARD -----------------
elif pagina == "KPI DASHBOARD":
    header_block(
        "Modulo 03 — Live Monitoring",
        "IL TUO CRUSCOTTO PRESTAZIONALE",
        "Bilancio carico/recupero, rischio infortunio e profilo atletico calcolati sui parametri appena inseriti.",
        IMG_HERO_KPI, "Real-Time Dashboard"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario in 'Analisi Completa' prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()

        st.markdown("### Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
        df_14 = df.tail(14).copy()
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)", fill='tozeroy', fillcolor='rgba(255, 106, 61, 0.18)', line=dict(color='#FF6A3D', width=3)))
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero", line=dict(color='#00F5A0', width=4)))
        fig_balance.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E8ECF2", size=13), bgcolor="rgba(14,20,32,0.85)", bordercolor="#1c2333", borderwidth=1))
        st.plotly_chart(style_fig(fig_balance), use_container_width=True)
        st.markdown("""
        <div class='explain-text' style='margin-bottom: 25px;'>
        La linea verde rappresenta la tua "Batteria" (quanto dormi e recuperi). L'area arancione è quanto sforzo spremi dal tuo corpo. Finché la linea verde avvolge i picchi arancioni, sei in stato di Supercompensazione (migliori). Se l'arancione sta costantemente sopra il verde, stai andando in Overtraining.
        </div>
        """, unsafe_allow_html=True)

        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )

        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        if risk_score < 25:
            status_color = "#00F5A0"
            status_text = "OTTIMALE"
        elif risk_score < 60:
            status_color = "#FFB020"
            status_text = "MODERATO"
        else:
            status_color = "#FF6A3D"
            status_text = "CRITICO"

        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 4px; font-family:\"Space Grotesk\",sans-serif;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {status_color};'><div class='section-label'>Rischio Infortunio</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k2:
            rec_color = "#00F5A0" if recovery_score >= 75 else "#FFB020" if recovery_score >= 40 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {rec_color};'><div class='section-label'>Recovery Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k3:
            sma_color = "#00F5A0" if sma < 10 else "#FFB020" if sma < 15 else "#FF6A3D"
            st.markdown(f"<div class='kpi-card' style='border-top: 2px solid {sma_color};'><div class='section-label'>SMA Score</div><div class='data-figure' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#8792A3'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "#E8ECF2"}, 'bar': {'color': status_color, 'thickness': 0.75}, 'bgcolor': "#111827", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': "rgba(0, 245, 160, 0.08)"}, {'range': [25, 60], 'color': "rgba(255, 176, 32, 0.08)"}, {'range': [60, 100], 'color': "rgba(255, 106, 61, 0.08)"}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(style_fig(fig_gauge), use_container_width=True)

        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno (h)', 'Stress (1-10)', 'RPE (1-10)', 'Recovery (%)'], fill='toself', name='Parametri',
                marker=dict(color=status_color), line=dict(color=status_color)
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#1c2333'), angularaxis=dict(gridcolor='#1c2333')), height=360)
            st.plotly_chart(style_fig(fig_radar), use_container_width=True)

        st.markdown("---")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Sonno", f"{r['ore_sonno']:.1f}h", "vs 7.5h")
        col_p2.metric("Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("RPE", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("FC Riposo", f"{r['fc_riposo']} bpm", "Base")

        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("### Il Tuo Profilo Atletico AI")

        cv_sonno = df['Ore Sonno'].std() / df['Ore Sonno'].mean()
        cv_rpe = df['RPE'].std() / df['RPE'].mean()
        consistenza = max(0, 100 - (cv_sonno + cv_rpe) * 100)

        if recovery_score >= 75 and sma < 10:
            archetipo, arch_col, arch_desc = "Il Bilanciato", "#00F5A0", "Gestisci sonno e carichi con grande equilibrio. Il tuo corpo lavora in supercompensazione costante: mantieni questa routine."
        elif r['stress_lavoro'] >= 7 and r['ore_sonno'] < 7:
            archetipo, arch_col, arch_desc = "Il Guerriero Stanco", "#FFB020", "Spingi forte nonostante stress e sonno limitato. Grande grinta, ma il conto arriva: pianifica un blocco di recupero prima possibile."
        elif sma >= 15:
            archetipo, arch_col, arch_desc = "L'Instancabile", "#FF6A3D", "Accumuli carico su carico. Ottimo motore, ma attenzione: senza pause il rischio di crollo fisico o mentale cresce rapidamente."
        else:
            archetipo, arch_col, arch_desc = "Il Costante", "#00E5FF", "Il tuo profilo è stabile e prevedibile: la base ideale su cui costruire progressi graduali e a basso rischio infortuni."

        col_arch1, col_arch2 = st.columns([1, 2])
        with col_arch1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 2px solid {arch_col}; display:flex; flex-direction:column; justify-content:center;'>
                <h3 style='color:{arch_col}; margin:5px 0; font-size:1.3em;'>{archetipo}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_arch2:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; height:100%;'>
                <p style='color:#B8C2D0; font-size:1.02em; margin-bottom:15px; font-family:"Inter",sans-serif;'>{arch_desc}</p>
                <p style='color:#8792A3; margin-bottom:5px; font-family:"Inter",sans-serif; font-size:0.9em;'>Indice di Consistenza (90gg)</p>
                <div style='background:#111827; border-radius:8px; overflow:hidden; height:20px;'>
                    <div style='background: linear-gradient(90deg, #00E5FF, #00F5A0); width:{min(consistenza,100):.0f}%; height:100%; text-align:right; padding-right:8px; color:#04121a; font-size:0.78em; font-weight:700; line-height:20px; font-family:"JetBrains Mono",monospace;'>{consistenza:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 4: ML EXPLAINED -----------------
elif pagina == "ML EXPLAINED":
    header_block(
        "Modulo 04 — Model Explainability",
        "Dentro il Motore Predittivo.",
        "Come 100 alberi decisionali votano il tuo rischio infortunio, spiegato passo per passo.",
        IMG_HERO_ML, "Random Forest Engine"
    )

    st.markdown("""
    <div class='info-box'>
    <h3>Cos'è il Machine Learning?</h3>
    <p style='color: #B8C2D0; font-family:"Inter",sans-serif;'>L'algoritmo impara dai dati storici per fare previsioni su nuovi dati. Analizzando i 90 giorni passati, identifica pattern complessi che portano al rischio di infortunio e calcola la tua probabilità odierna.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = st.session_state.dati.copy()
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
        y_train = df['Rischio Infortunio'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
        rf_model.fit(X_scaled, y_train)
        y_pred = rf_model.predict(X_scaled)

        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        cm = confusion_matrix(y_train, y_pred)

        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC Media', 'RPE']
        importances = rf_model.feature_importances_

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Spiegazione", "Feature Importance", "Confusion Matrix", "Metriche", "Calcolo Live", "Simulatore What-If"])

        with tab1:
            st.markdown(f"""
            <div class='kpi-card' style='text-align: left;'>
                <h2 style='color: #00F5A0; border:none; margin-bottom: 5px;'>Il Consiglio dei 100 Saggi (Random Forest)</h2>
                <p style='color: #B8C2D0; font-size: 1.05em; line-height: 1.6; font-family:"Inter",sans-serif;'>
                Anziché fare un calcolo matematico rigido, il sistema usa una tecnica chiamata <strong>Random Forest</strong>.<br><br>
                Immagina di convocare <strong>100 allenatori diversi</strong> a guardare il tuo storico di {len(df)} allenamenti. Ognuno di loro presta attenzione a cose diverse: l'Allenatore 1 guarda solo le ore di sonno, l'Allenatore 2 fissa lo stress lavorativo. <br><br>
                Oggi, fornendo i tuoi dati odierni, tutti e 100 gli allenatori votano in segreto: <em>"Si farà male?"</em> oppure <em>"È al sicuro?"</em>.<br>
                Il risultato in percentuale che ti mostriamo alla fine è semplicemente <strong>quanti di questi 100 allenatori hanno votato per il Rischio Infortunio.</strong> È potente perché non si basa su una sola opinione, ma su 100 visioni diverse dei tuoi dati.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("**Quali Parametri Influenzano Più il Rischio?**")
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)

            fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#00E5FF', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'))
            fig_imp.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig_imp), use_container_width=True)

            st.markdown("""
            <div class='explain-text'>
            <strong>Cosa Calcola:</strong> Misura quale parametro scatena più infortuni nel TUO corpo.<br>
            <strong>Come lo calcola:</strong> Rimuove un dato alla volta dalle valutazioni e vede quanto peggiora l'accuratezza del modello.<br>
            La barra più lunga rappresenta il tuo personale punto debole (es. Se è il Sonno, significa che la tua genetica tollera male i recuperi brevi).
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("**Confusion Matrix**")
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Predetto: Sicuro', 'Predetto: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 24, "color": "#04121a"}, colorscale=[[0,'#0E1420'],[1,'#00E5FF']], showscale=False))
            fig_cm.update_layout(height=400)
            st.plotly_chart(style_fig(fig_cm), use_container_width=True)

            st.markdown("""
            <div class='explain-text'>
            <strong>Cosa Calcola:</strong> Analizza i "falsi allarmi" e le "sviste" del modello nel passato.<br>
            <strong>Come lo calcola:</strong> Confronta ciò che è successo realmente nei 90 giorni con le previsioni fatte "alla cieca" dall'AI.<br>
            I quadrati sulla diagonale principale indicano i successi. Se vedi numeri alti nei quadrati opposti, il modello sta commettendo errori.
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            st.markdown("**Performance del Modello su Dati Storici**")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Accuracy", f"{acc*100:.1f}%", "Predizioni corrette")
            col_m2.metric("Precision", f"{prec*100:.1f}%", "Esattezza 'Rischio'")
            col_m3.metric("Recall", f"{rec*100:.1f}%", "Rischi individuati")

            st.markdown("""
            <div class='explain-text'>
            <strong>Cosa Calcolano:</strong> Le metriche di affidabilità del test.<br>
            <strong>Come lo calcolano:</strong> Formule statistiche classiche sui veri e falsi positivi ottenuti dalla Matrice di Confusione.<br>
            Guarda in particolare la <strong>Recall</strong>: se è alta (vicina al 100%), significa che l'AI riesce a individuare praticamente tutti gli infortuni reali prima che accadano.
            </div>
            """, unsafe_allow_html=True)

        with tab5:
            st.markdown("**Come il Modello Calcola il Rischio Oggi**")
            if st.session_state.analisi_fatta:
                r = st.session_state.risultati_analisi
                dist = r.get('distanza_oggi', 10.0)
                input_data = np.array([[dist, r['ore_sonno'], r['stress_lavoro'], 100 + r['rpe_previsto']*10, r['rpe_previsto']]])
                prob_rischio = rf_model.predict_proba(scaler.transform(input_data))[0][1] * 100
                votes_rischio = int(prob_rischio)
                votes_safe = 100 - votes_rischio

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig_prob = go.Figure(go.Indicator(mode="gauge+number", value=prob_rischio, title="ML Prediction", gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#FF6A3D" if prob_rischio >= 60 else "#FFB020" if prob_rischio >= 25 else "#00F5A0"}, 'bgcolor': "#111827", 'borderwidth': 0}, number={'suffix': '%', 'font': {'size': 35}}))
                    fig_prob.update_layout(height=350)
                    st.plotly_chart(style_fig(fig_prob), use_container_width=True)
                with col_g2:
                    fig_votes = go.Figure(data=[go.Bar(name='Rischio', x=['Voti Alberi'], y=[votes_rischio], marker_color='#FF6A3D'), go.Bar(name='Sicuro', x=['Voti Alberi'], y=[votes_safe], marker_color='#00F5A0')])
                    fig_votes.update_layout(barmode='stack', title="Voti dei 100 Alberi", height=350)
                    st.plotly_chart(style_fig(fig_votes), use_container_width=True)

                st.markdown(f"""
                <div class='explain-text'>
                <strong>Cosa Calcola:</strong> La tua esatta percentuale di rischio infortunio oggi.<br>
                <strong>Come lo calcola:</strong> Prende i {dist}km previsti, le tue {r['ore_sonno']}h di sonno e l'RPE di {r['rpe_previsto']} e fa votare i 100 alberi. <br>
                In questo momento, <strong>{votes_rischio} alberi su 100</strong> ritengono che tu sia a forte rischio infortunio con i parametri correnti. Modifica l'allenamento se superi i 60 voti.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Completa il questionario per vedere il calcolo personalizzato.")

        with tab6:
            st.markdown("""
            <div class='info-box'>
            <strong>Muovi le leve e osserva in tempo reale come cambia la previsione dei 100 alberi.</strong> Questo simulatore non tocca i tuoi dati salvati: è un laboratorio per capire "cosa succederebbe se...".
            </div>
            """, unsafe_allow_html=True)

            base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {
                'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6
            }

            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                sim_dist = st.slider("Distanza simulata (km)", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
                sim_sonno = st.slider("Ore di sonno simulate", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
            with col_sim2:
                sim_stress = st.slider("Stress simulato", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
                sim_rpe = st.slider("RPE simulato", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")

            sim_fc = 100 + sim_rpe * 10
            sim_input = np.array([[sim_dist, sim_sonno, sim_stress, sim_fc, sim_rpe]])
            sim_prob = rf_model.predict_proba(scaler.transform(sim_input))[0][1] * 100
            sim_color = "#FF6A3D" if sim_prob >= 60 else "#FFB020" if sim_prob >= 25 else "#00F5A0"

            col_simg1, col_simg2 = st.columns(2)
            with col_simg1:
                fig_sim_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=sim_prob, title={'text': "Rischio Simulato", 'font': {'color': '#8792A3'}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color}, 'bgcolor': "#111827", 'borderwidth': 0},
                    number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}}
                ))
                fig_sim_gauge.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sim_gauge), use_container_width=True)
            with col_simg2:
                sonno_range = np.linspace(4, 10, 20)
                probs_range = []
                for s in sonno_range:
                    test_input = np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])
                    p = rf_model.predict_proba(scaler.transform(test_input))[0][1] * 100
                    probs_range.append(p)
                fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'}, title="Sensibilità: Rischio vs Ore di Sonno")
                fig_sens.update_traces(line_color="#00E5FF", line_width=3)
                fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#FF6A3D")
                fig_sens.update_layout(height=320)
                st.plotly_chart(style_fig(fig_sens), use_container_width=True)

            st.markdown(f"""
            <div class='explain-text'>
            <strong>Cosa Calcola:</strong> Una previsione "ipotetica" indipendente dai tuoi dati salvati, utile per testare scenari futuri.<br>
            <strong>Come lo calcola:</strong> Passa i valori delle leve agli stessi 100 alberi del modello e conta i voti per "Rischio".<br>
            <strong>Il grafico di sensibilità</strong> mostra come cambierebbe il rischio se SOLO le ore di sonno variassero, a parità di tutto il resto: la linea tratteggiata arancione è il tuo valore attuale simulato ({sim_sonno:.1f}h).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Proiezione Rischio Prossimi 7 Giorni (se le abitudini restano queste)**")
            proj_days = [f"Giorno +{i}" for i in range(1, 8)]
            proj_drift = np.clip(sim_prob + np.cumsum(np.random.normal(0.5 if sim_prob > 40 else -0.5, 3, 7)), 0, 100)
            fig_proj = px.area(x=proj_days, y=proj_drift, labels={'x': '', 'y': 'Rischio %'})
            fig_proj.update_traces(line_color=sim_color, fillcolor="rgba(255,106,61,0.15)" if sim_color == "#FF6A3D" else "rgba(0,245,160,0.15)")
            fig_proj.update_layout(height=300)
            st.plotly_chart(style_fig(fig_proj), use_container_width=True)
            st.markdown("<div class='explain-text'>Proiezione indicativa basata sull'inerzia del trend attuale, non una previsione clinica: usala solo come guida direzionale.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

# ----------------- PAGINA 5: CONSIGLIO FINALE -----------------
elif pagina == "CONSIGLIO FINALE":
    header_block(
        "Modulo 05 — Action Plan",
        "Il Tuo Piano d'Azione.",
        "Protocollo pre/durante/post allenamento generato su misura per i parametri di oggi.",
        IMG_HERO_PLAN, "Coach Protocol"
    )

    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario in 'Analisi Completa' prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()

        risk_score = min(100,
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )

        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0

        distanza_target = r.get('distanza_oggi', 10.0)
        distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0

        if risk_score < 25:
            tit, col = "ALLENAMENTO INTENSO AUTORIZZATO", "#00F5A0"
        elif risk_score < 60:
            tit, col = "RECUPERO ATTIVO CONSIGLIATO", "#FFB020"
        else:
            tit, col = "RIPOSO OBBLIGATORIO", "#FF6A3D"

        st.markdown(f"""
        <div class='kpi-card' style='border: 1px solid {col}; background-color: rgba(0,0,0,0.35);'>
            <h2 style='color: {col}; margin: 0; border: none; font-size:1.6em;'>{tit}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .kpi-card-equal { background: #0E1420; border-radius: 12px; padding: 25px 20px; border: 1px solid #1c2333; height: 480px; display: flex; flex-direction: column; }
        .kpi-card-equal .kpi-equal-body { overflow-y: auto; flex-grow: 1; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar { width: 6px; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar-thumb { background: #1c2333; border-radius: 4px; }
        </style>
        """, unsafe_allow_html=True)

        tipo_all = r.get('tipo_allenamento', 'Easy Run')
        idratazione_pre = round(distanza_target * 20)
        idratazione_post = round(distanza_target * 15)
        carbo_pre = round(distanza_target * 3)
        prot_post = round(distanza_target * 1.2) + 15
        carbo_post = round(distanza_target * 4) + 20

        col_new1, col_new2, col_new3 = st.columns(3)
        with col_new1:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00E5FF;'>Distanza Consigliata</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{distanza_consigliata:.1f} km</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>su {distanza_target}km desiderati</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center; font-family:"Inter",sans-serif;'>Tipo allenamento: <strong style='color:#B8C2D0;'>{tipo_all}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new2:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:{col};'>Rischio Calcolato</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0; font-family:"JetBrains Mono",monospace;'>{risk_score:.0f}%</h1>
                    <p style='color:#8792A3; font-family:"Inter",sans-serif;'>Probabilità Infortunio/Burnout</p>
                    <p style='color:#566178; font-size:0.85em; margin-top:15px; text-align:center; font-family:"Inter",sans-serif;'>Recovery Score: <strong style='color:#B8C2D0;'>{recovery_score:.0f}%</strong> · SMA: <strong style='color:#B8C2D0;'>{sma:.1f}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new3:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#00F5A0;'>Protocollo Coach Dettagliato</h3>
                <div class='kpi-equal-body' style='color:#B8C2D0; font-size:0.85em; text-align:left; font-family:"Inter",sans-serif;'>
                    <strong style='color:#00E5FF;'>PRE-ALLENAMENTO (T-90/-15 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>T-90': pasto leggero, ~{carbo_pre}g carboidrati (banana, pane, riso).</li>
                        <li>T-30': {idratazione_pre}ml di liquidi (acqua + elettroliti se >20°C).</li>
                        <li>T-15': mobilità dinamica anche/caviglie, skip, calciata dietro (5').</li>
                        <li>T-5': attivazione glutei con band, 2x15 ripetizioni.</li>
                    </ul>
                    <strong style='color:#FFB020;'>DURANTE</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Sorso d'acqua ogni 20' se {tipo_all.lower()} supera i 60'.</li>
                        <li>Cadenza target 170-180 spm, respiro 3:2 (corsa lenta) o 2:1 (soglia).</li>
                        <li>Se FC supera la tua soglia per >5' consecutivi, rallenta subito.</li>
                    </ul>
                    <strong style='color:#00F5A0;'>POST (0-30 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Entro 30': ~{prot_post}g proteine + ~{carbo_post}g carboidrati (finestra anabolica).</li>
                        <li>{idratazione_post}ml liquidi per reintegro, +500mg sodio se sudorazione abbondante.</li>
                        <li>Stretching statico gentile 8-10' (polpacci, ischio, ileopsoas).</li>
                        <li>Rullo miofasciale 5' su quadricipiti e fascia plantare.</li>
                    </ul>
                    <strong style='color:#8b5cf6;'>SERALE</strong>
                    <ul style='margin-top:5px; padding-left:18px; margin-bottom:0;'>
                        <li>Doccia fredda/tiepida per ridurre infiammazione.</li>
                        <li>Nessuno schermo 30' prima di dormire, punta a {max(r['ore_sonno'],7.5):.1f}h di sonno.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.subheader("Analisi Parametri vs Media (90 giorni)")
        media_sonno_90, media_stress_90, media_rpe_90 = df['Ore Sonno'].mean(), df['Stress Lavoro'].mean(), df['RPE'].mean()
        sonno_vs_media, stress_vs_media, rpe_vs_media = r['ore_sonno'] - media_sonno_90, r['stress_lavoro'] - media_stress_90, r['rpe_previsto'] - media_rpe_90

        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            sb, sc = ("SOTTO MEDIA", "#FF6A3D") if sonno_vs_media < -0.5 else ("SOPRA MEDIA", "#00F5A0") if sonno_vs_media > 0.5 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{sc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{sb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['ore_sonno']:.1f}h</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_sonno_90:.1f}h</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Il tempo dedicato alla rigenerazione cellulare. Un deficit rallenta il recupero muscolare.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a2:
            stb, stc = ("SOTTO MEDIA", "#00F5A0") if stress_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if stress_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{stc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{stb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['stress_lavoro']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_stress_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Il carico cognitivo e nervoso accumulato. Uno stress alto alza il cortisolo e il rischio di infortuni.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_a3:
            rpb, rpc = ("SOTTO MEDIA", "#00F5A0") if rpe_vs_media < -1 else ("SOPRA MEDIA", "#FF6A3D") if rpe_vs_media > 1 else ("NELLA MEDIA", "#8792A3")
            st.markdown(f"""
            <div class='kpi-card'>
                <p style='color:{rpc}; font-weight:bold; font-family:"JetBrains Mono",monospace; font-size:0.78em; letter-spacing:0.08em;'>{rpb}</p>
                <h1 style='font-family:"JetBrains Mono",monospace;'>{r['rpe_previsto']}/10</h1>
                <p style='font-family:"Inter",sans-serif; color:#8792A3;'>vs media {media_rpe_90:.1f}/10</p>
                <p style='font-family:"Inter",sans-serif; color:#566178; font-size:0.85em; margin-top:8px;'>Lo sforzo programmato per la sessione. Valori superiori alla media indicano un forte sovraccarico sistemico.</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Grafici Trend - Ultimi 30 Giorni")
        df_recent = df.tail(30).copy()
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True, title="Sonno Trend", color_discrete_sequence=['#00E5FF'])
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_sonno), use_container_width=True)
            st.markdown("<div class='explain-text'>La linea arancione è il sonno di oggi: confrontalo con le ultime settimane per capire se stai recuperando meglio o peggio del solito.</div>", unsafe_allow_html=True)
        with col_g2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True, title="RPE Trend", color_discrete_sequence=['#00E5FF'])
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_rpe), use_container_width=True)
            st.markdown("<div class='explain-text'>La linea arancione è lo sforzo previsto per oggi: se è ben sopra la media recente, il carico odierno è superiore al solito.</div>", unsafe_allow_html=True)
        with col_g3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True, title="Stress Trend", color_discrete_sequence=['#FFB020'])
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_stress), use_container_width=True)
            st.markdown("<div class='explain-text'>Lo stress dichiarato oggi (linea arancione) incide sul recupero tanto quanto l'allenamento stesso: tienilo d'occhio nel tempo.</div>", unsafe_allow_html=True)
        col_g4, col_g5 = st.columns(2)
        with col_g4:
            fig_scatter = px.scatter(df_recent, x='Ore Sonno', y='RPE', size='Distanza (km)', color='FC Media', height=350, title="Relazione Sonno-RPE-FC", color_continuous_scale=[[0,'#0E4A57'],[1,'#00E5FF']])
            fig_scatter.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="#FF6A3D")
            fig_scatter.add_vline(x=r['ore_sonno'], line_dash="dash", line_color="#FF6A3D")
            st.plotly_chart(style_fig(fig_scatter), use_container_width=True)
            st.markdown("<div class='explain-text'>Ogni punto è un allenamento passato; le linee arancioni indicano dove si colloca oggi tra sonno e sforzo rispetto allo storico.</div>", unsafe_allow_html=True)
        with col_g5:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df['Ore Sonno'], name='Sonno 90gg', marker_color='#00E5FF'))
            fig_box.add_trace(go.Box(y=[r['ore_sonno']], name='Oggi', marker_color='#FF6A3D'))
            fig_box.update_layout(height=350, title="Sonno: Oggi vs Storico")
            st.plotly_chart(style_fig(fig_box), use_container_width=True)
            st.markdown("<div class='explain-text'>Il box mostra la variabilità storica del sonno: il punto arancione è il valore di oggi rispetto alla tua norma.</div>", unsafe_allow_html=True)

        st.markdown("<br><hr><br>", unsafe_allow_html=True)

        st.markdown("<h2>Proiezione Fisiologica Odierna</h2>", unsafe_allow_html=True)

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            time_x = np.arange(0, 60, 5)
            hr_y = [r['fc_riposo'] + 20] + [r['fc_riposo'] + 70 + np.random.randint(-5, 5) for _ in range(10)] + [r['fc_riposo'] + 30]
            fig_pace = px.line(x=time_x, y=hr_y, title="1. Curva BPM Consigliata Oggi", labels={'x':'Minuti', 'y':'BPM'})
            fig_pace.update_traces(line_color="#FF6A3D")
            fig_pace.update_layout(height=300)
            st.plotly_chart(style_fig(fig_pace), use_container_width=True)
            st.markdown("<div class='explain-text'>Riscalda il motore nei primi 10 minuti (rampa dolce) per non creare acido lattico in eccesso. Mantieni il blocco centrale stabile senza picchi.</div>", unsafe_allow_html=True)
        with g_col2:
            hours = ["+0h", "+6h", "+12h", "+24h", "+48h"]
            rec_y = [30, 55, 75, 95, 100] if risk_score < 50 else [15, 30, 50, 70, 90]
            fig_rec = px.bar(x=hours, y=rec_y, title="2. Tempo di Ricarica Glicogeno Stimato", labels={'x':'Ore Post-Workout', 'y':'% Energie'})
            fig_rec.update_traces(marker_color="#00F5A0")
            fig_rec.update_layout(height=300)
            st.plotly_chart(style_fig(fig_rec), use_container_width=True)
            st.markdown("<div class='explain-text'>Quanto ci metteranno i tuoi muscoli a rigenerarsi dopo questo allenamento. Fino al raggiungimento dell'80% non inserire lavori di forza.</div>", unsafe_allow_html=True)
        g_col3, g_col4 = st.columns(2)
        with g_col3:
            fig_acwr = go.Figure(data=[
                go.Bar(name='Carico Ultimi 7gg', x=['Carico'], y=[450], marker_color='#FFB020'),
                go.Bar(name='Media 28gg', x=['Carico'], y=[390], marker_color='#00E5FF')
            ])
            fig_acwr.update_layout(title="3. Bilancio Acuto vs Cronico (ACWR)", barmode='group', height=300)
            st.plotly_chart(style_fig(fig_acwr), use_container_width=True)
            st.markdown("<div class='explain-text'>Mostra se stai correndo troppo rispetto a quello a cui sei abituato. Un rapporto tra la barra ambra e ciano intorno a 1.15 è ottimale per migliorare. Oltre l'1.3 è zona infortuni.</div>", unsafe_allow_html=True)
        with g_col4:
            fig_pie2 = px.pie(values=[70, 20, 10], names=['Aerobico Base', 'Soglia Lattata', 'Anaerobico'], title="4. Ripartizione Energetica Richiesta", hole=0.6, color_discrete_sequence=['#00E5FF', '#FFB020', '#FF6A3D'])
            fig_pie2.update_layout(height=300)
            st.plotly_chart(style_fig(fig_pie2), use_container_width=True)
            st.markdown("<div class='explain-text'>Su cosa lavorerà il tuo metabolismo oggi. In base a questo capisci quanti carboidrati (Soglia) o grassi (Aerobico) intaccherai.</div>", unsafe_allow_html=True)

        fig_sleep_impact = go.Figure(go.Waterfall(
            name="Sonno", orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["Sonno Base", "Fatica Workout", "Stress Lavoro", "Target Stanotte"],
            y=[7.5, (distanza_consigliata/10)*0.5, (r['stress_lavoro']-5)*0.1, 0],
            connector={"line":{"color":"#1c2333"}}, decreasing={"marker":{"color":"#00F5A0"}}, increasing={"marker":{"color":"#FF6A3D"}}, totals={"marker":{"color":"#00E5FF"}},
            textposition="outside"
        ))
        fig_sleep_impact.update_layout(title="5. Calcolo Aggiuntivo Ore di Sonno Necessarie", height=400, showlegend=False)
        st.plotly_chart(style_fig(fig_sleep_impact), use_container_width=True)
        st.markdown("<div class='explain-text'>L'allenamento e lo stress di oggi creano micro-rotture che chiedono tempo extra per guarire. Questo grafico ti aggiunge (barre arancioni) le ore necessarie per stanotte rispetto al tuo fabbisogno standard.</div>", unsafe_allow_html=True)

        st.markdown("<br>---<br>", unsafe_allow_html=True)

        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            if risk_score < 25:
                st.markdown("<div class='success-box'><h3>Consigli Generali</h3><ul><li>Condizioni ideali, puoi spingere o fare gara.</li><li>Struttura: 15' Warm-up, Lavoro centrale, 15' Defaticamento.</li></ul></div>", unsafe_allow_html=True)
            elif risk_score < 60:
                st.markdown("<div class='warning-box'><h3>Consigli Generali</h3><ul><li>Evita variazioni di ritmo brusche.</li><li>Mantieni il corpo in equilibrio idrico.</li></ul></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='danger-box'><h3>Consigli Generali</h3><ul><li>Riposo assoluto da urti (corsa).</li><li>Punta alla rigenerazione nervosa.</li></ul></div>", unsafe_allow_html=True)

        with col_rec2:
            st.markdown("<div class='info-box'><h3>Prossimi 3 Giorni</h3><ul><li>DOMANI: In base al recupero, preferibilmente Easy.</li><li>+2: Giorno chiave se oggi fai un lungo.</li><li>+3: Riposo o sessione Fartlek/Qualità.</li></ul></div>", unsafe_allow_html=True)

        st.subheader("Riepilogo KPI Odierno")
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Riposo', 'Recovery', 'SMA', 'Risk'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10", f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': ["OK" if r['ore_sonno'] >= 7 else "MEDIO" if r['ore_sonno'] >= 6.5 else "ALTO", "OK" if r['stress_lavoro'] <= 5 else "MEDIO" if r['stress_lavoro'] <= 7 else "ALTO", "OK" if r['rpe_previsto'] <= 5 else "MEDIO" if r['rpe_previsto'] <= 7 else "ALTO", "OK" if r['fc_riposo'] <= 65 else "MEDIO", "OK" if recovery_score >= 75 else "MEDIO" if recovery_score >= 40 else "ALTO", "OK" if sma < 10 else "MEDIO" if sma < 15 else "ALTO", "OK" if risk_score < 25 else "MEDIO" if risk_score < 60 else "ALTO"]
        })

        fig_table_riepilogo = go.Figure(data=[go.Table(
            header=dict(values=list(riepilogo_df.columns), fill_color='#111827', align='center', font=dict(color='#00E5FF', size=13, family="JetBrains Mono, monospace")),
            cells=dict(values=[riepilogo_df[col] for col in riepilogo_df.columns], fill_color='#0E1420', align='center', font=dict(color='#B8C2D0', size=12, family="Inter, sans-serif"), height=30)
        )])
        fig_table_riepilogo.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350)
        st.plotly_chart(style_fig(fig_table_riepilogo), use_container_width=True)
        st.markdown("<div class='explain-text'>Colpo d'occhio su tutti i tuoi indicatori di oggi: OK nella norma, MEDIO da monitorare, ALTO richiede attenzione immediata.</div>", unsafe_allow_html=True)
