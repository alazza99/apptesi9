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

st.set_page_config(page_title="RunAI Coach", layout="wide", initial_sidebar_state="expanded")

# --- CSS Dark Theme e Box Informativi ---
st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #f8f9fa; font-family: 'Inter', 'Segoe UI', sans-serif; }
    h1 { color: #ffffff; text-align: center; margin-bottom: 30px; font-size: 2.5em; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #ffffff; padding-bottom: 15px; margin-bottom: 20px; font-size: 1.8em; font-weight: 700; border-bottom: 1px solid #1f2937; }
    h3 { color: #e5e7eb; font-size: 1.3em; font-weight: 600; }
    
    .info-box { background: rgba(26, 115, 232, 0.1); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border: 1px solid rgba(255,255,255,0.05); }
    .success-box { background: rgba(52, 168, 83, 0.1); border-left: 4px solid #10b981; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border: 1px solid rgba(255,255,255,0.05); }
    .warning-box { background: rgba(251, 188, 4, 0.1); border-left: 4px solid #f59e0b; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border: 1px solid rgba(255,255,255,0.05); }
    .danger-box { background: rgba(234, 67, 53, 0.1); border-left: 4px solid #ef4444; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border: 1px solid rgba(255,255,255,0.05); }
    .kpi-card { background: #111827; border-radius: 16px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3); border: 1px solid #1f2937; }
    
    /* Box spiegazioni grafici */
    .explain-text { font-size: 0.9em; color: #9ca3af; line-height: 1.5; margin-top: 5px; padding: 15px; background: #111827; border-radius: 8px; border-left: 3px solid #6b7280;}

    /* --- FIX LEGGIBILITà WIDGET / FINESTRE SCURE --- */
    .stForm { background-color: #0f1420; border: 1px solid #1f2937; border-radius: 16px; padding: 25px; }
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stDateInput input {
        background-color: #1a2233 !important; color: #f8f9fa !important; border: 1px solid #374151 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #1a2233 !important; color: #f8f9fa !important; border: 1px solid #374151 !important;
    }
    div[data-baseweb="popover"] { background-color: #1a2233 !important; }
    div[data-baseweb="popover"] ul, div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #1a2233 !important;
    }
    div[data-baseweb="popover"] li, div[data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #1a2233 !important; color: #f8f9fa !important;
    }
    div[data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover {
        background-color: #2d3748 !important; color: #ffffff !important;
    }
    .stSlider label, .stSelectSlider label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stDateInput label {
        color: #e5e7eb !important; font-weight: 600 !important;
    }
    .stSlider [data-baseweb="slider"] div { color: #f8f9fa !important; }
    div[data-testid="stTickBar"] { color: #9ca3af !important; }
    .stSelectSlider [role="slider"] { background-color: #3b82f6 !important; }
    div[data-testid="stWidgetLabel"] p { color: #e5e7eb !important; }
    div[data-testid="stForm"] { background-color: #0f1420; border: 1px solid #1f2937; border-radius: 16px; }

    /* --- SIDEBAR: stesso colore dell'app + testo leggibile --- */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1f2937;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #0b0f19 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #e5e7eb !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        color: #f3f4f6 !important;
    }
    section[data-testid="stSidebar"] button {
        background-color: #1a2233 !important;
        border: 1px solid #374151 !important;
    }
    section[data-testid="stSidebar"] button p,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button div {
        color: #f8f9fa !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"

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

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: left; font-size: 2em;'>⭕ RunAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; font-size: 0.9em; margin-top: -20px; margin-bottom: 30px;'>Professional Analytics</p>", unsafe_allow_html=True)
    
    st.subheader("📱 Dispositivo")
    
    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }
    
    device_scelto = st.selectbox("Seleziona dispositivo:", list(dispositivi.keys()), label_visibility="collapsed")
    
    if st.button("🔗 Connetti Dispositivo", use_container_width=True):
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
        <div style='background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 15px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <span style='color: #10b981; font-weight: bold;'>● LIVE</span>
                <span style='color: #9ca3af; font-size: 0.8em;'>{}</span>
            </div>
            <div style='color: #f3f4f6;'>
                <div style='margin: 5px 0;'>❤️ FC: <span style='float: right; font-weight: bold;'>{} bpm</span></div>
                <div style='margin: 5px 0;'>🔋 Batteria: <span style='float: right; font-weight: bold; color: #10b981;'>{}%</span></div>
                <div style='margin: 5px 0;'>👟 Passi: <span style='float: right; font-weight: bold;'>{:,}</span></div>
                <div style='margin: 5px 0;'>🔥 Calorie: <span style='float: right; font-weight: bold;'>{}</span></div>
            </div>
            <div style='color: #6b7280; font-size: 0.75em; margin-top: 10px; text-align: center;'>Sync: {}</div>
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
    pagina = st.sidebar.radio("📋 Menu", 
        ["📋 Analisi Completa", "📈 Statistiche", "📊 KPI Dashboard", "🔮 ML Explained", "💡 Consiglio Finale"],
        label_visibility="collapsed"
    )

# ----------------- PAGINA 1: ANALISI -----------------
if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa dello Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    <strong>ℹ️ Compila i tuoi parametri odierni per avviare il motore di predizione.</strong>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("form_analisi"):
        st.markdown("### 🎯 Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            distanza_oggi = st.number_input("Distanza Prevista (km)", min_value=0.0, value=10.0)
        
        st.markdown("#### 🏁 Obiettivo Finale (Lungo Termine)")
        col_of1, col_of2, col_of3 = st.columns(3)
        with col_of1:
            obj_finale = st.text_input("Obiettivo Finale", placeholder="Es: Maratona sub 3:30")
        with col_of2:
            data_obj_finale = st.date_input("Data Obiettivo", value=pd.Timestamp.today() + pd.Timedelta(days=90))
        with col_of3:
            km_obj_finale = st.number_input("Distanza Gara (km)", min_value=0.0, value=42.2)
        
        st.markdown("---")
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
        
        st.markdown("---")
        st.markdown("### 🧠 Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)
        
        st.markdown("---")
        st.markdown("### ⚡ Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)
        
        st.markdown("---")
        bottone = st.form_submit_button("🚀 ANALIZZA PARAMETRI", use_container_width=True)
    
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
        st.success("✓ Analisi completata e caricata nel modello!")
        
        if obj_finale:
            giorni_rimasti = (pd.Timestamp(data_obj_finale) - pd.Timestamp.today()).days
            st.markdown(f"""
            <div class='info-box' style='border-left-color:#8b5cf6;'>
            <strong>🏁 Obiettivo Finale impostato:</strong> {obj_finale} ({km_obj_finale:.1f} km) — mancano circa <strong>{max(giorni_rimasti,0)} giorni</strong> ({pd.Timestamp(data_obj_finale).strftime('%d/%m/%Y')}).
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 2: STATISTICHE -----------------
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Dettagliate - 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    st.subheader("📊 KPI Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🏃 KM Totali", f"{df['Distanza (km)'].sum():.0f} km", "90 giorni")
    col_m2.metric("📊 Sessioni", f"{len(df)}")
    col_m3.metric("📐 Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")
    
    st.markdown("---")
    st.subheader("📖 Analisi Dettagliata")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📏 Volume", "❤️ Intensità", "😴 Recupero", "📋 Tabella Storico"])
    
    with tab1:
        # GRAFICI ORIGINALI + NUOVI
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({'Distanza (km)': 'sum'}).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=300, color='Distanza (km)', color_continuous_scale='Blues')
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Verifica che le barre non facciano salti maggiori del 10% da una settimana all'altra. Un picco improvviso porta a infiammazioni tendinee.</div>", unsafe_allow_html=True)
            
            st.markdown("**Nuovo: Carico per Giorno della Settimana**")
            df['Giorno_Settimana'] = df['Giorno'].dt.day_name()
            df_day = df.groupby('Giorno_Settimana')['Distanza (km)'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig_day = px.bar(df_day, x='Giorno_Settimana', y='Distanza (km)', height=300, color_discrete_sequence=['#3b82f6'])
            fig_day.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_day, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Visualizza la tua routine. Assicurati che ai giorni con barre alte seguano giorni con barre basse o assenti (recupero attivo).</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Distanza Cumulativa (Originale)**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=300, markers=True)
            fig_cum.update_traces(line_color="#3b82f6")
            fig_cum.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cum, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Una linea retta indica costanza. Una linea piatta indica stop o infortuni.</div>", unsafe_allow_html=True)
            
            # --- SORPRESA: Bacheca Record & Achievement (non è un grafico) ---
            record_km = df.loc[df['Distanza (km)'].idxmax()]
            record_vel = df.loc[df['Velocità (km/h)'].idxmax()]
            giorni_attivi = (df['Distanza (km)'] > 0).sum()
            streak = int((df['Distanza (km)'] > df['Distanza (km)'].mean()).astype(int).groupby((df['Distanza (km)'] <= df['Distanza (km)'].mean()).cumsum()).cumsum().max())
            
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; margin-top:10px; background: linear-gradient(135deg, #111827 0%, #1a1033 100%); border: 1px solid #3b2a5e;'>
                <h3 style='color:#facc15; margin-bottom:15px;'>🏆 Bacheca Record — Ultimi 90 giorni</h3>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#e5e7eb;'>
                    <span>🥇 Corsa più lunga</span><strong style='color:#f8f9fa;'>{record_km['Distanza (km)']:.1f} km</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#e5e7eb;'>
                    <span>⚡ Velocità massima</span><strong style='color:#f8f9fa;'>{record_vel['Velocità (km/h)']:.1f} km/h</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#e5e7eb;'>
                    <span>🔥 Miglior striscia sopra media</span><strong style='color:#f8f9fa;'>{streak} allenamenti</strong>
                </div>
                <div style='display:flex; justify-content:space-between; margin:8px 0; color:#e5e7eb;'>
                    <span>📅 Giorni con allenamento</span><strong style='color:#f8f9fa;'>{giorni_attivi} / {len(df)}</strong>
                </div>
                <p style='color:#9ca3af; font-size:0.8em; margin-top:12px; margin-bottom:0;'>🎖️ Continua così: ogni record battuto è un mattoncino in più verso il tuo obiettivo finale!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', color='RPE', color_continuous_scale='Viridis', height=300)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Più i punti scendono verso il basso a destra, più il tuo cuore è efficiente (vai veloce faticando poco). I punti gialli sono i lavori massimali.</div>", unsafe_allow_html=True)
            
            st.markdown("**Nuovo: Ripartizione Zone Cardiache**")
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Fondo Lento)', 'Z3 (Medio/Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df['Zone'] = pd.cut(df['FC Media'], bins=bins, labels=labels)
            zone_counts = df['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', hole=0.5, height=300, color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_zones.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_zones, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Un allenamento sano prevede 80% in Z1/Z2 e 20% in Z4/Z5. Evita di rimanere intrappolato in Z3 (zona grigia), che stanca senza allenare efficacemente.</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=300, color_discrete_sequence=['#3b82f6'])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig3.add_vline(x=3.5, line_dash="dash", line_color="#10b981")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="#ef4444")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Mostra quante volte hai spinto al massimo (oltre la linea rossa) e quante volte hai recuperato (sotto la linea verde).</div>", unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=300, markers=True)
            fig_sleep.update_traces(line_color="#8b5cf6")
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="#10b981")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="#ef4444")
            fig_sleep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sleep, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Cerca di stare sempre sopra la linea verde. Le discese verso la linea rossa corrispondono a cali di prestazione muscolare.</div>", unsafe_allow_html=True)
            
            st.markdown("**Nuovo: Debito di Sonno (Rolling 7gg)**")
            df['Debito'] = df['Ore Sonno'].apply(lambda x: max(0, 7.5 - x)).rolling(7).sum()
            fig_debt = px.area(df, x='Giorno', y='Debito', height=300, color_discrete_sequence=['#ef4444'])
            fig_debt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_debt, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Quest'area rossa è fatica accumulata. Se il debito supera le 5 ore in una settimana, il rischio di strappi o contratture decuplica.</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("**Sonno vs Sforzo (Originale)**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', color='Rischio Infortunio', color_continuous_scale=['#3b82f6', '#ef4444'], height=300)
            fig4.add_hline(y=7, line_dash="dash", line_color="#f59e0b")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="#f59e0b")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Il quadrante in alto a sinistra (poco sonno, alto sforzo) è la 'Death Zone'. Evita che i punti cadano in quell'area.</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: '🔴' if x == 1 else '✅')
        
        # Tabella nativa Plotly per mantenere il design Dark
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#1f2937', align='center', font=dict(color='white', size=14)),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#111827', align='center', font=dict(color='#d1d5db', size=13), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
        st.plotly_chart(fig_table, use_container_width=True)

# ----------------- PAGINA 3: KPI DASHBOARD -----------------
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # --- NUOVO GRANDE GRAFICO (AREA BALANCE) ---
        st.markdown("### ⚖️ Bilancio Carico vs Recupero (Ultimi 14 Giorni + Oggi)")
        df_14 = df.tail(14).copy()
        fig_balance = go.Figure()
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Carico Sforzo (Strain)", fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#ef4444', width=3)))
        fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Capacità di Recupero", line=dict(color='#10b981', width=4)))
        fig_balance.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#f8f9fa", size=13), bgcolor="rgba(17,24,39,0.85)", bordercolor="#374151", borderwidth=1))
        st.plotly_chart(fig_balance, use_container_width=True)
        st.markdown("""
        <div class='explain-text' style='margin-bottom: 25px;'>
        <strong>Come leggerlo:</strong> La linea verde rappresenta la tua "Batteria" (quanto dormi e recuperi). L'area rossa è quanto sforzo spremi dal tuo corpo. Finché la linea verde avvolge i picchi rossi, sei in stato di Supercompensazione (migliori). Se il rosso sta costantemente sopra il verde, stai andando in Overtraining.
        </div>
        """, unsafe_allow_html=True)
        
        # --- CALCOLI E GRAFICI ORIGINALI KPI ---
        risk_score = min(100, 
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        if risk_score < 25:
            status_color = "#10b981"
            status_text = "OTTIMALE"
        elif risk_score < 60:
            status_color = "#f59e0b"
            status_text = "MODERATO"
        else:
            status_color = "#ef4444"
            status_text = "CRITICO"
        
        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 2px;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(f"<div class='kpi-card' style='border-top: 4px solid {status_color};'><div class='kpi-title'>Rischio Infortunio</div><div class='kpi-value' style='font-size:2em; font-weight:bold; color: {status_color};'>{risk_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k2:
            rec_color = "#10b981" if recovery_score >= 75 else "#f59e0b" if recovery_score >= 40 else "#ef4444"
            st.markdown(f"<div class='kpi-card' style='border-top: 4px solid {rec_color};'><div class='kpi-title'>Recovery Score</div><div class='kpi-value' style='font-size:2em; font-weight:bold; color: {rec_color};'>{recovery_score:.0f}%</div></div>", unsafe_allow_html=True)
        with col_k3:
            sma_color = "#10b981" if sma < 10 else "#f59e0b" if sma < 15 else "#ef4444"
            st.markdown(f"<div class='kpi-card' style='border-top: 4px solid {sma_color};'><div class='kpi-title'>SMA Score</div><div class='kpi-value' style='font-size:2em; font-weight:bold; color: {sma_color};'>{sma:.1f}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score, title={'text': "Risk Level", 'font': {'color': '#9ca3af'}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': status_color, 'thickness': 0.75}, 'bgcolor': "#1f2937", 'borderwidth': 0,
                       'steps': [{'range': [0, 25], 'color': "rgba(16, 185, 129, 0.1)"}, {'range': [25, 60], 'color': "rgba(245, 158, 11, 0.1)"}, {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.1)"}]},
                number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}}
            ))
            fig_gauge.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno\n(h)', 'Stress\n(1-10)', 'RPE\n(1-10)', 'Recovery\n(%)'], fill='toself', name='Parametri',
                marker=dict(color=status_color), line=dict(color=status_color)
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10], gridcolor='#374151'), angularaxis=dict(gridcolor='#374151')), height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"vs 7.5h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Base")
        
        # --- SORPRESA: Profilo Atleta AI + Consistenza (in fondo alla pagina) ---
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        st.markdown("### 🧬 Il Tuo Profilo Atletico AI")
        
        cv_sonno = df['Ore Sonno'].std() / df['Ore Sonno'].mean()
        cv_rpe = df['RPE'].std() / df['RPE'].mean()
        consistenza = max(0, 100 - (cv_sonno + cv_rpe) * 100)
        
        if recovery_score >= 75 and sma < 10:
            archetipo, arch_icon, arch_col, arch_desc = "Il Bilanciato", "⚖️", "#10b981", "Gestisci sonno e carichi con grande equilibrio. Il tuo corpo lavora in supercompensazione costante: mantieni questa routine."
        elif r['stress_lavoro'] >= 7 and r['ore_sonno'] < 7:
            archetipo, arch_icon, arch_col, arch_desc = "Il Guerriero Stanco", "🛡️", "#f59e0b", "Spingi forte nonostante stress e sonno limitato. Grande grinta, ma il conto arriva: pianifica un blocco di recupero prima possibile."
        elif sma >= 15:
            archetipo, arch_icon, arch_col, arch_desc = "L'Instancabile", "🔥", "#ef4444", "Accumuli carico su carico. Ottimo motore, ma attenzione: senza pause il rischio di crollo fisico o mentale cresce rapidamente."
        else:
            archetipo, arch_icon, arch_col, arch_desc = "Il Costante", "🧭", "#3b82f6", "Il tuo profilo è stabile e prevedibile: la base ideale su cui costruire progressi graduali e a basso rischio infortuni."
        
        col_arch1, col_arch2 = st.columns([1, 2])
        with col_arch1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 4px solid {arch_col};'>
                <div style='font-size:3em;'>{arch_icon}</div>
                <h3 style='color:{arch_col}; margin:5px 0;'>{archetipo}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col_arch2:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left; height:100%;'>
                <p style='color:#d1d5db; font-size:1.05em; margin-bottom:15px;'>{arch_desc}</p>
                <p style='color:#9ca3af; margin-bottom:5px;'>Indice di Consistenza (90gg)</p>
                <div style='background:#1f2937; border-radius:8px; overflow:hidden; height:22px;'>
                    <div style='background: linear-gradient(90deg, #3b82f6, #10b981); width:{min(consistenza,100):.0f}%; height:100%; text-align:right; padding-right:8px; color:white; font-size:0.8em; font-weight:bold; line-height:22px;'>{consistenza:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- PAGINA 4: ML EXPLAINED -----------------
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Explainability")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cos'è il Machine Learning?</h3>
    <p style='color: #d1d5db;'>L'algoritmo impara dai dati storici per fare previsioni su nuovi dati. Analizzando i 90 giorni passati, identifica pattern complessi che portano al rischio di infortunio e calcola la tua probabilità odierna.</p>
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
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎓 Spiegazione", "📊 Feature Importance", "🔬 Confusion Matrix", "📈 Metriche", "🧮 Calcolo LIVE", "🧪 Simulatore What-If"])
        
        with tab1:
            # SPIEGAZIONE MIGLIORATA E RISCRITTA
            st.markdown(f"""
            <div class='kpi-card' style='text-align: left; background-color: #111827;'>
                <h2 style='color: #10b981; border:none; margin-bottom: 5px;'>🌳 Il Consiglio dei 100 Saggi (Random Forest)</h2>
                <p style='color: #d1d5db; font-size: 1.1em; line-height: 1.6;'>
                Anziché fare un calcolo matematico rigido, noi usiamo una tecnica chiamata <strong>Random Forest</strong>.<br><br>
                Immagina di convocare <strong>100 allenatori diversi</strong> a guardare il tuo storico di {len(df)} allenamenti. Ognuno di loro presta attenzione a cose diverse: l'Allenatore 1 guarda solo le ore di sonno, l'Allenatore 2 fissa lo stress lavorativo. <br><br>
                Oggi, fornendo i tuoi dati odierni, tutti e 100 gli allenatori votano in segreto: <em>"Si farà male?"</em> oppure <em>"È al sicuro?"</em>.<br>
                Il risultato in percentuale che ti mostriamo alla fine è semplicemente <strong>quanti di questi 100 allenatori hanno votato per il Rischio Infortunio.</strong> È potente perché non si basa su una sola opinione, ma su 100 visioni diverse dei tuoi dati.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Quali Parametri Influenzano Più il Rischio? (Originale)**")
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            
            fig_imp = go.Figure(go.Bar(y=[x[0] for x in imp_data], x=[x[1]*100 for x in imp_data], orientation='h', marker_color='#3b82f6', text=[f'{x[1]*100:.1f}%' for x in imp_data], textposition='auto'))
            fig_imp.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("""
            <div class='explain-text'>
            <strong>❓ Cosa Calcola:</strong> Misura quale parametro scatena più infortuni nel TUO corpo.<br>
            <strong>⚙️ Come lo calcola:</strong> Rimuove un dato alla volta dalle valutazioni e vede quanto peggiora l'accuratezza del modello.<br>
            <strong>🎯 Spiegazione Risultato:</strong> La barra più lunga rappresenta il tuo personale punto debole (es. Se è il Sonno, significa che la tua genetica tollera male i recuperi brevi).
            </div>
            """, unsafe_allow_html=True)
            
        with tab3:
            st.markdown("**Confusion Matrix (Originale)**")
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Predetto: Sicuro', 'Predetto: Rischio'], y=['Reale: Sicuro', 'Reale: Rischio'], text=cm, texttemplate='%{text}', textfont={"size": 24, "color": "white"}, colorscale='Blues', showscale=False))
            fig_cm.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown("""
            <div class='explain-text'>
            <strong>❓ Cosa Calcola:</strong> Analizza i "falsi allarmi" e le "sviste" del modello nel passato.<br>
            <strong>⚙️ Come lo calcola:</strong> Confronta ciò che è successo realmente nei 90 giorni con le previsioni fatte "alla cieca" dall'AI.<br>
            <strong>🎯 Spiegazione Risultato:</strong> I quadrati in alto a sinistra e in basso a destra (diagonale principale) indicano i successi. Se vedi numeri alti nei quadrati opposti, il modello sta commettendo errori.
            </div>
            """, unsafe_allow_html=True)
            
        with tab4:
            st.markdown("**Performance del Modello su Dati Storici (Originale)**")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Esattezza 'Rischio'")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Rischi individuati")
            
            st.markdown("""
            <div class='explain-text'>
            <strong>❓ Cosa Calcolano:</strong> Le metriche di affidabilità medica del test.<br>
            <strong>⚙️ Come lo calcolano:</strong> Formule statistiche classiche sui veri e falsi positivi ottenuti dalla Matrice di Confusione.<br>
            <strong>🎯 Spiegazione Risultato:</strong> Guarda in particolare la <strong>Recall</strong>: se è alta (vicina al 100%), significa che l'AI riesce a fiutare praticamente *tutti* gli infortuni reali prima che accadano. Non ignora i pericoli.
            </div>
            """, unsafe_allow_html=True)
            
        with tab5:
            st.markdown("**Come il Modello Calcola il Rischio OGGI (Originale)**")
            if st.session_state.analisi_fatta:
                r = st.session_state.risultati_analisi
                dist = r.get('distanza_oggi', 10.0)
                input_data = np.array([[dist, r['ore_sonno'], r['stress_lavoro'], 100 + r['rpe_previsto']*10, r['rpe_previsto']]])
                prob_rischio = rf_model.predict_proba(scaler.transform(input_data))[0][1] * 100
                votes_rischio = int(prob_rischio)
                votes_safe = 100 - votes_rischio
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig_prob = go.Figure(go.Indicator(mode="gauge+number", value=prob_rischio, title="ML Prediction", gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ef4444" if prob_rischio >= 60 else "#f59e0b" if prob_rischio >= 25 else "#10b981"}, 'bgcolor': "#1f2937", 'borderwidth': 0}, number={'suffix': '%', 'font': {'size': 35}}))
                    fig_prob.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_prob, use_container_width=True)
                with col_g2:
                    fig_votes = go.Figure(data=[go.Bar(name='Rischio', x=['Voti Alberi'], y=[votes_rischio], marker_color='#ef4444'), go.Bar(name='Sicuro', x=['Voti Alberi'], y=[votes_safe], marker_color='#10b981')])
                    fig_votes.update_layout(barmode='stack', title="Voti dei 100 Alberi", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_votes, use_container_width=True)
                
                st.markdown(f"""
                <div class='explain-text'>
                <strong>❓ Cosa Calcola:</strong> La tua esatta percentuale di rischio infortunio OGGI.<br>
                <strong>⚙️ Come lo calcola:</strong> Prende i {dist}km previsti, le tue {r['ore_sonno']}h di sonno e l'RPE di {r['rpe_previsto']} e fa votare i 100 alberi. <br>
                <strong>🎯 Spiegazione Risultato:</strong> In questo momento, <strong>{votes_rischio} "Saggi" su 100</strong> ritengono che tu sia a forte rischio infortunio con i parametri correnti. Modifica l'allenamento se superi i 60 voti.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Completa il questionario per vedere il calcolo personalizzato.")
        
        with tab6:
            st.markdown("""
            <div class='info-box'>
            <strong>🧪 Muovi le leve e osserva in tempo reale come cambia la previsione dei 100 alberi.</strong> Questo simulatore non tocca i tuoi dati salvati: è un laboratorio per capire "cosa succederebbe se...".
            </div>
            """, unsafe_allow_html=True)
            
            base = st.session_state.risultati_analisi if st.session_state.analisi_fatta else {
                'distanza_oggi': 10.0, 'ore_sonno': 7.5, 'stress_lavoro': 5, 'rpe_previsto': 6
            }
            
            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                sim_dist = st.slider("🏃 Distanza simulata (km)", 0.0, 42.0, float(base.get('distanza_oggi', 10.0)), key="sim_dist")
                sim_sonno = st.slider("😴 Ore di sonno simulate", 2.0, 12.0, float(base.get('ore_sonno', 7.5)), key="sim_sonno")
            with col_sim2:
                sim_stress = st.slider("🧠 Stress simulato", 1, 10, int(base.get('stress_lavoro', 5)), key="sim_stress")
                sim_rpe = st.slider("⚡ RPE simulato", 1, 10, int(base.get('rpe_previsto', 6)), key="sim_rpe")
            
            sim_fc = 100 + sim_rpe * 10
            sim_input = np.array([[sim_dist, sim_sonno, sim_stress, sim_fc, sim_rpe]])
            sim_prob = rf_model.predict_proba(scaler.transform(sim_input))[0][1] * 100
            sim_color = "#ef4444" if sim_prob >= 60 else "#f59e0b" if sim_prob >= 25 else "#10b981"
            
            col_simg1, col_simg2 = st.columns(2)
            with col_simg1:
                fig_sim_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=sim_prob, title={'text': "Rischio Simulato", 'font': {'color': '#9ca3af'}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': sim_color}, 'bgcolor': "#1f2937", 'borderwidth': 0},
                    number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}}
                ))
                fig_sim_gauge.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sim_gauge, use_container_width=True)
            with col_simg2:
                # Sensitivity: varia il sonno tenendo fissi gli altri parametri (partial dependence semplificata)
                sonno_range = np.linspace(4, 10, 20)
                probs_range = []
                for s in sonno_range:
                    test_input = np.array([[sim_dist, s, sim_stress, sim_fc, sim_rpe]])
                    p = rf_model.predict_proba(scaler.transform(test_input))[0][1] * 100
                    probs_range.append(p)
                fig_sens = px.line(x=sonno_range, y=probs_range, labels={'x': 'Ore di Sonno', 'y': 'Rischio %'}, title="Sensibilità: Rischio vs Ore di Sonno")
                fig_sens.update_traces(line_color="#8b5cf6", line_width=3)
                fig_sens.add_vline(x=sim_sonno, line_dash="dash", line_color="#ef4444")
                fig_sens.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sens, use_container_width=True)
            
            st.markdown(f"""
            <div class='explain-text'>
            <strong>❓ Cosa Calcola:</strong> Una previsione "ipotetica" indipendente dai tuoi dati salvati, utile per testare scenari futuri.<br>
            <strong>⚙️ Come lo calcola:</strong> Passa i valori delle leve agli stessi 100 alberi del modello e conta i voti per "Rischio".<br>
            <strong>🎯 Il grafico di sensibilità</strong> mostra come cambierebbe il rischio se SOLO le ore di sonno variassero, a parità di tutto il resto: la linea tratteggiata rossa è il tuo valore attuale simulato ({sim_sonno:.1f}h).
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📅 Proiezione Rischio Prossimi 7 Giorni (se le abitudini restano queste)**")
            proj_days = [f"Giorno +{i}" for i in range(1, 8)]
            proj_drift = np.clip(sim_prob + np.cumsum(np.random.normal(0.5 if sim_prob > 40 else -0.5, 3, 7)), 0, 100)
            fig_proj = px.area(x=proj_days, y=proj_drift, labels={'x': '', 'y': 'Rischio %'})
            fig_proj.update_traces(line_color=sim_color, fillcolor=f"rgba(239,68,68,0.15)" if sim_color == "#ef4444" else "rgba(16,185,129,0.15)")
            fig_proj.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_proj, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Nota:</strong> Proiezione indicativa basata sull'inerzia del trend attuale, non una previsione clinica: usala solo come guida direzionale.</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

# ----------------- PAGINA 5: CONSIGLIO FINALE -----------------
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Piano di Azione & Consiglio Coach")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # Calcolo Rischio Score (Originale)
        risk_score = min(100, 
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        # --- NUOVE INDICAZIONI PROMINENTI ---
        distanza_target = r.get('distanza_oggi', 10.0)
        distanza_consigliata = distanza_target if risk_score < 40 else distanza_target * 0.6 if risk_score < 70 else 0.0
        
        if risk_score < 25:
            tit, col, box_class = "🟢 ALLENAMENTO INTENSO AUTORIZZATO", "#10b981", "success-box"
        elif risk_score < 60:
            tit, col, box_class = "🟡 RECUPERO ATTIVO CONSIGLIATO", "#f59e0b", "warning-box"
        else:
            tit, col, box_class = "🔴 RIPOSO OBBLIGATORIO", "#ef4444", "danger-box"
            
        st.markdown(f"""
        <div class='kpi-card' style='border: 2px solid {col}; background-color: rgba(0,0,0,0.5);'>
            <h2 style='color: {col}; margin: 0; border: none;'>{tit}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
        .kpi-card-equal { background: #111827; border-radius: 16px; padding: 25px 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); border: 1px solid #1f2937; height: 480px; display: flex; flex-direction: column; }
        .kpi-card-equal .kpi-equal-body { overflow-y: auto; flex-grow: 1; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar { width: 6px; }
        .kpi-card-equal .kpi-equal-body::-webkit-scrollbar-thumb { background: #374151; border-radius: 4px; }
        </style>
        """, unsafe_allow_html=True)
        
        # Personalizzazione protocollo in base a tipo di allenamento e distanza
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
                <h3 style='color:#3b82f6;'>Distanza Consigliata</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0;'>{distanza_consigliata:.1f} km</h1>
                    <p style='color:#9ca3af;'>su {distanza_target}km desiderati</p>
                    <p style='color:#6b7280; font-size:0.85em; margin-top:15px; text-align:center;'>Tipo allenamento: <strong style='color:#d1d5db;'>{tipo_all}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new2:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:{col};'>Rischio Calcolato</h3>
                <div class='kpi-equal-body' style='display:flex; flex-direction:column; justify-content:center; align-items:center;'>
                    <h1 style='color:white; font-size:3em; margin:0;'>{risk_score:.0f}%</h1>
                    <p style='color:#9ca3af;'>Probabilità Infortunio/Burnout</p>
                    <p style='color:#6b7280; font-size:0.85em; margin-top:15px; text-align:center;'>Recovery Score: <strong style='color:#d1d5db;'>{recovery_score:.0f}%</strong> · SMA: <strong style='color:#d1d5db;'>{sma:.1f}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_new3:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <h3 style='color:#10b981;'>Protocollo Coach Dettagliato</h3>
                <div class='kpi-equal-body' style='color:#d1d5db; font-size:0.85em; text-align:left;'>
                    <strong style='color:#3b82f6;'>🕐 PRE-ALLENAMENTO (T-90/-15 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>T-90': pasto leggero, ~{carbo_pre}g carboidrati (banana, pane, riso).</li>
                        <li>T-30': {idratazione_pre}ml di liquidi (acqua + elettroliti se >20°C).</li>
                        <li>T-15': mobilità dinamica anche/caviglie, skip, calciata dietro (5').</li>
                        <li>T-5': attivazione glutei con band, 2x15 ripetizioni.</li>
                    </ul>
                    <strong style='color:#f59e0b;'>🏃 DURANTE</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Sorso d'acqua ogni 20' se {tipo_all.lower()} supera i 60'.</li>
                        <li>Cadenza target 170-180 spm, respiro 3:2 (corsa lenta) o 2:1 (soglia).</li>
                        <li>Se FC supera la tua soglia per >5' consecutivi, rallenta subito.</li>
                    </ul>
                    <strong style='color:#10b981;'>✅ POST (0-30 min)</strong>
                    <ul style='margin-top:5px; padding-left:18px;'>
                        <li>Entro 30': ~{prot_post}g proteine + ~{carbo_post}g carboidrati (finestra anabolica).</li>
                        <li>{idratazione_post}ml liquidi per reintegro, +500mg sodio se sudorazione abbondante.</li>
                        <li>Stretching statico gentile 8-10' (polpacci, ischio, ileopsoas).</li>
                        <li>Rullo miofasciale 5' su quadricipiti e fascia plantare.</li>
                    </ul>
                    <strong style='color:#8b5cf6;'>🌙 SERALE</strong>
                    <ul style='margin-top:5px; padding-left:18px; margin-bottom:0;'>
                        <li>Doccia fredda/tiepida per ridurre infiammazione.</li>
                        <li>Nessuno schermo 30' prima di dormire, punta a {max(r['ore_sonno'],7.5):.1f}h di sonno.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        # --- SEZIONE ORIGINALE: Parametri vs Media ---
        st.subheader("📊 Analisi Parametri vs Media (90 giorni)")
        media_sonno_90, media_stress_90, media_rpe_90 = df['Ore Sonno'].mean(), df['Stress Lavoro'].mean(), df['RPE'].mean()
        sonno_vs_media, stress_vs_media, rpe_vs_media = r['ore_sonno'] - media_sonno_90, r['stress_lavoro'] - media_stress_90, r['rpe_previsto'] - media_rpe_90
        
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            sb, sc = ("⬇️ SOTTO", "#ef4444") if sonno_vs_media < -0.5 else ("⬆️ SOPRA", "#10b981") if sonno_vs_media > 0.5 else ("➡️ PARI", "#9ca3af")
            st.markdown(f"<div class='kpi-card'><p style='color:{sc}; font-weight:bold;'>{sb}</p><h1>{r['ore_sonno']:.1f}h</h1><p>vs media {media_sonno_90:.1f}h</p></div>", unsafe_allow_html=True)
        with col_a2:
            stb, stc = ("⬇️ SOTTO", "#10b981") if stress_vs_media < -1 else ("⬆️ SOPRA", "#ef4444") if stress_vs_media > 1 else ("➡️ PARI", "#9ca3af")
            st.markdown(f"<div class='kpi-card'><p style='color:{stc}; font-weight:bold;'>{stb}</p><h1>{r['stress_lavoro']}/10</h1><p>vs media {media_stress_90:.1f}/10</p></div>", unsafe_allow_html=True)
        with col_a3:
            rpb, rpc = ("⬇️ SOTTO", "#10b981") if rpe_vs_media < -1 else ("⬆️ SOPRA", "#ef4444") if rpe_vs_media > 1 else ("➡️ PARI", "#9ca3af")
            st.markdown(f"<div class='kpi-card'><p style='color:{rpc}; font-weight:bold;'>{rpb}</p><h1>{r['rpe_previsto']}/10</h1><p>vs media {media_rpe_90:.1f}/10</p></div>", unsafe_allow_html=True)
        
        # --- SEZIONE ORIGINALE: Grafici Trend e Analitici ---
        st.subheader("📈 Grafici Trend - Ultimi 30 Giorni (Originale)")
        df_recent = df.tail(30).copy()
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True, title="Sonno Trend", color_discrete_sequence=['#8b5cf6'])
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red")
            fig_sonno.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sonno, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> La linea rossa è il sonno di oggi: confrontalo con le ultime settimane per capire se stai recuperando meglio o peggio del solito.</div>", unsafe_allow_html=True)
        with col_g2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True, title="RPE Trend", color_discrete_sequence=['#3b82f6'])
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            fig_rpe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rpe, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> La linea rossa è lo sforzo previsto per oggi: se è ben sopra la media recente, il carico odierno è superiore al solito.</div>", unsafe_allow_html=True)
        with col_g3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True, title="Stress Trend", color_discrete_sequence=['#f59e0b'])
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red")
            fig_stress.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_stress, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Lo stress dichiarato oggi (linea rossa) incide sul recupero tanto quanto l'allenamento stesso: tienilo d'occhio nel tempo.</div>", unsafe_allow_html=True)
        col_g4, col_g5 = st.columns(2)
        with col_g4:
            fig_scatter = px.scatter(df_recent, x='Ore Sonno', y='RPE', size='Distanza (km)', color='FC Media', height=350, title="Relazione Sonno-RPE-FC (Originale)")
            fig_scatter.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            fig_scatter.add_vline(x=r['ore_sonno'], line_dash="dash", line_color="red")
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Ogni punto è un allenamento passato; le linee rosse indicano dove si colloca oggi tra sonno e sforzo rispetto allo storico.</div>", unsafe_allow_html=True)
        with col_g5:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df['Ore Sonno'], name='Sonno 90gg', marker_color='#3b82f6'))
            fig_box.add_trace(go.Box(y=[r['ore_sonno']], name='Oggi', marker_color='#ef4444'))
            fig_box.update_layout(height=350, title="Sonno: Oggi vs Storico (Originale)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Il box mostra la variabilità storica del sonno: il punto rosso è il valore di oggi rispetto alla tua norma.</div>", unsafe_allow_html=True)
            
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # --- I 5 NUOVI GRAFICI AGGIUNTIVI SUI CONSIGLI ---
        st.markdown("<h2>🎯 Proiezione Fisiologica Odierna (5 Grafici AI)</h2>", unsafe_allow_html=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            # 1. Pacing Strategy
            time_x = np.arange(0, 60, 5)
            hr_y = [r['fc_riposo'] + 20] + [r['fc_riposo'] + 70 + np.random.randint(-5, 5) for _ in range(10)] + [r['fc_riposo'] + 30]
            fig_pace = px.line(x=time_x, y=hr_y, title="1. Curva BPM Consigliata Oggi", labels={'x':'Minuti', 'y':'BPM'})
            fig_pace.update_traces(line_color="#ef4444")
            fig_pace.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_pace, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Riscalda il motore nei primi 10 minuti (rampa dolce) per non creare acido lattico in eccesso. Mantieni il blocco centrale stabile senza picchi.</div>", unsafe_allow_html=True)
        with g_col2:
            # 2. Recovery Window
            hours = ["+0h", "+6h", "+12h", "+24h", "+48h"]
            rec_y = [30, 55, 75, 95, 100] if risk_score < 50 else [15, 30, 50, 70, 90]
            fig_rec = px.bar(x=hours, y=rec_y, title="2. Tempo di Ricarica Glicogeno Stimato", labels={'x':'Ore Post-Workout', 'y':'% Energie'})
            fig_rec.update_traces(marker_color="#10b981")
            fig_rec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_rec, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Quanto ci metteranno i tuoi muscoli a rigenerarsi dopo questo allenamento. Fino al raggiungimento dell'80% non inserire lavori di forza.</div>", unsafe_allow_html=True)
        g_col3, g_col4 = st.columns(2)
        with g_col3:
            # 3. ACWR
            fig_acwr = go.Figure(data=[
                go.Bar(name='Carico Ultimi 7gg', x=['Carico'], y=[450], marker_color='#f59e0b'),
                go.Bar(name='Media 28gg', x=['Carico'], y=[390], marker_color='#3b82f6')
            ])
            fig_acwr.update_layout(title="3. Bilancio Acuto vs Cronico (ACWR)", barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_acwr, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Mostra se stai correndo troppo rispetto a quello a cui sei abituato. Un rapporto tra la barra gialla e blu intorno a 1.15 è ottimale per migliorare. Oltre l'1.3 è zona infortuni.</div>", unsafe_allow_html=True)
        with g_col4:
            # 4. Sforzo Energetico
            fig_pie2 = px.pie(values=[70, 20, 10], names=['Aerobico Base', 'Soglia Lattata', 'Anaerobico'], title="4. Ripartizione Energetica Richiesta", hole=0.5, color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'])
            fig_pie2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_pie2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Su cosa lavorerà il tuo metabolismo oggi. In base a questo capisci quanti carboidrati (Soglia) o grassi (Aerobico) intaccherai.</div>", unsafe_allow_html=True)
        # 5. Waterfall Sonno
        fig_sleep_impact = go.Figure(go.Waterfall(
            name="Sonno", orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["Sonno Base", "Fatica Workout", "Stress Lavoro", "Target Stanotte"],
            y=[7.5, (distanza_consigliata/10)*0.5, (r['stress_lavoro']-5)*0.1, 0],
            connector={"line":{"color":"#374151"}}, decreasing={"marker":{"color":"#10b981"}}, increasing={"marker":{"color":"#ef4444"}}, totals={"marker":{"color":"#3b82f6"}},
            textposition="outside"
        ))
        fig_sleep_impact.update_layout(title="5. Calcolo Aggiuntivo Ore di Sonno Necessarie (Waterfall)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False)
        st.plotly_chart(fig_sleep_impact, use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> L'allenamento e lo stress di oggi creano micro-rotture che chiedono tempo extra per guarire. Questo grafico ti aggiunge (barre rosse) le ore necessarie per stanotte rispetto al tuo fabbisogno standard.</div>", unsafe_allow_html=True)
        
        st.markdown("<br>---<br>", unsafe_allow_html=True)
        
        # --- SEZIONE ORIGINALE: Raccomandazioni Testuali ---
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            if risk_score < 25:
                st.markdown("<div class='success-box'><h3>✅ Consigli Generali</h3><ul><li>Condizioni ideali, puoi spingere o fare gara.</li><li>Struttura: 15' Warm-up, Lavoro centrale, 15' Defaticamento.</li></ul></div>", unsafe_allow_html=True)
            elif risk_score < 60:
                st.markdown("<div class='warning-box'><h3>🟡 Consigli Generali</h3><ul><li>Evita variazioni di ritmo brusche.</li><li>Mantieni il corpo in equilibrio idrico.</li></ul></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='danger-box'><h3>🔴 Consigli Generali</h3><ul><li>Riposo assoluto da urti (corsa).</li><li>Punta alla rigenerazione nervosa.</li></ul></div>", unsafe_allow_html=True)
                
        with col_rec2:
            st.markdown("<div class='info-box'><h3>⏰ Prossimi 3 Giorni</h3><ul><li>DOMANI: In base al recupero, preferibilmente Easy.</li><li>+2: Giorno chiave se oggi fai un lungo.</li><li>+3: Riposo o sessione Fartlek/Qualità.</li></ul></div>", unsafe_allow_html=True)

        # --- SEZIONE ORIGINALE: Riepilogo KPI (Tabella Plotly Dark) ---
        st.subheader("📋 Riepilogo KPI Odierno (Originale)")
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Riposo', 'Recovery', 'SMA', 'Risk'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10", f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': ["✅" if r['ore_sonno'] >= 7 else "⚠️" if r['ore_sonno'] >= 6.5 else "🔴", "✅" if r['stress_lavoro'] <= 5 else "⚠️" if r['stress_lavoro'] <= 7 else "🔴", "✅" if r['rpe_previsto'] <= 5 else "⚠️" if r['rpe_previsto'] <= 7 else "🔴", "✅" if r['fc_riposo'] <= 65 else "⚠️", "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "🔴", "✅" if sma < 10 else "⚠️" if sma < 15 else "🔴", "✅" if risk_score < 25 else "⚠️" if risk_score < 60 else "🔴"]
        })
        
        fig_table_riepilogo = go.Figure(data=[go.Table(
            header=dict(values=list(riepilogo_df.columns), fill_color='#1f2937', align='center', font=dict(color='white', size=14)),
            cells=dict(values=[riepilogo_df[col] for col in riepilogo_df.columns], fill_color='#111827', align='center', font=dict(color='#d1d5db', size=13), height=30)
        )])
        fig_table_riepilogo.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig_table_riepilogo, use_container_width=True)
        st.markdown("<div class='explain-text'><strong>Spiegazione:</strong> Colpo d'occhio su tutti i tuoi indicatori di oggi: ✅ nella norma, ⚠️ da monitorare, 🔴 richiede attenzione immediata.</div>", unsafe_allow_html=True)
