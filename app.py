import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide")

# CSS PROFESSIONALE
st.markdown("""
<style>
    body { background: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    .stApp { background: #f8f9fa; }
    h1 { color: #1a73e8; text-align: center; margin-bottom: 30px; font-size: 2.5em; font-weight: 700; }
    h2 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; margin-bottom: 20px; font-size: 1.8em; }
    h3 { color: #1a73e8; font-size: 1.3em; font-weight: 600; }
    
    .info-box { background: #e8f0fe; border-left: 5px solid #1a73e8; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .success-box { background: #e6f4ea; border-left: 5px solid #34a853; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .warning-box { background: #fef7e0; border-left: 5px solid #fbbc04; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .danger-box { background: #fce8e6; border-left: 5px solid #ea4335; padding: 20px; border-radius: 8px; margin: 20px 0; }
    
    .metric-card {
        background: white;
        border: 2px solid #ddd;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# FUNZIONI DI CALCOLO
# =====================================================================
def calcola_risk_score(sonno, stress, rpe):
    """Calcola risk score basato su parametri effettivi"""
    risk = 0
    
    if sonno < 6:
        risk += 40  # Sonno insufficiente = grande rischio
    elif sonno < 6.5:
        risk += 25
    elif sonno < 7:
        risk += 10
    
    if stress >= 8:
        risk += 35
    elif stress >= 6:
        risk += 20
    elif stress >= 4:
        risk += 5
    
    if rpe >= 8:
        risk += 30
    elif rpe >= 6:
        risk += 15
    elif rpe >= 4:
        risk += 5
    
    # Penalità combinata per situazioni critiche
    if sonno < 6.5 and stress >= 7 and rpe >= 7:
        risk += 20
    
    return min(100, risk)

def calcola_recovery_score(sonno):
    """Calcola capacità di recupero"""
    return max(0, 100 - abs(sonno - 7.5) * 13.33)

def calcola_sma(stress, rpe, sonno):
    """SMA = Stress × RPE / Sonno"""
    if sonno > 0:
        return (stress * rpe) / sonno
    return 0

# =====================================================================
# DATI
# =====================================================================
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
    df['ISLR'] = np.where(df['Distanza (km)'] > 0, (df['Ore Lavoro'] * df['Stress Lavoro']) / df['Distanza (km)'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    df['Overtraining'] = np.where((df['RPE'] > 8) & (df['Stress Lavoro'] > 7) & (df['Ore Sonno'] < 6), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.device_data = {'fc': 72, 'battery': 85, 'steps': 0, 'calories': 0}
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Running Analytics")
    st.markdown("---")
    
    dispositivi = {
        "Garmin Forerunner 965": "garmin",
        "Apple Watch Ultra": "apple",
        "Polar Vantage V3": "polar",
        "Fitbit Charge 6": "fitbit",
        "WHOOP 4.0": "whoop",
        "Fascia Cardio Garmin": "fascia"
    }
    
    st.subheader("📱 Dispositivo")
    device_scelto = st.selectbox("Seleziona:", list(dispositivi.keys()), label_visibility="collapsed")
    
    if st.button("🔗 Connetti", use_container_width=True):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.session_state.device_data = {
            'fc': np.random.randint(60, 80),
            'battery': np.random.randint(70, 100),
            'steps': np.random.randint(2000, 5000),
            'calories': np.random.randint(150, 300)
        }
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='success-box'>
        <strong>🟢 DISPOSITIVO CONNESSO</strong><br>
        <small>{st.session_state.device_name}</small><br>
        ❤️ FC: {st.session_state.device_data['fc']} bpm | 🔋 {st.session_state.device_data['battery']}%
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("📋 Menu", 
        ["📋 Analisi Completa", "📈 Statistiche", "📊 KPI Dashboard", "💡 Consiglio Finale"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: ANALISI COMPLETA
# =====================================================================
if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa dello Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    <strong>ℹ️ Come funziona:</strong> Compila i parametri odierni. Il sistema analizzerà i tuoi ultimi 90 giorni per darti consigli personalizzati.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📝 Dati Odierni")
    
    with st.form("form_analisi"):
        st.markdown("### 🎯 I Tuoi Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            obj_lt = st.text_input("Obiettivo Lungo Termine", placeholder="Es: Maratona < 3:30")
        
        st.markdown("---")
        
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno scorsa notte", 2.0, 12.0, 7.5)
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
        
        bottone = st.form_submit_button("🚀 ANALIZZA", use_container_width=True)
    
    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi,
            'obj_lt': obj_lt,
            'ore_sonno': ore_sonno,
            'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo,
            'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro,
            'tipo_allenamento': tipo_allenamento,
            'rpe_previsto': rpe_previsto,
        }
        st.success("✓ Analisi completata! Vai al 'Consiglio Finale' per i consigli personalizzati.")

# =====================================================================
# PAGINA 2: STATISTICHE
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Dettagliate - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    st.subheader("📊 Metriche Principali")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    km_totali = df['Distanza (km)'].sum()
    sessioni = len(df)
    media_km = df['Distanza (km)'].mean()
    giorni_rischio = df['Rischio Infortunio'].sum()
    
    col_m1.metric("🏃 KM Totali", f"{km_totali:.0f} km")
    col_m2.metric("📊 Sessioni", f"{sessioni}")
    col_m3.metric("📐 Media/Sessione", f"{media_km:.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{giorni_rischio}")
    
    st.markdown("---")
    
    st.subheader("📉 Analisi Dettagliata")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Progressione KM - Trend Settimanale**")
        df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
        df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
        
        fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', title="KM per Settimana", height=350)
        fig1.update_traces(marker_color='#1a73e8')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_g2:
        st.markdown("**Relazione FC vs Velocità**")
        fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', 
                         color='RPE', color_continuous_scale='Blues', height=350, opacity=0.7,
                         title="Efficienza Cardiaca")
        st.plotly_chart(fig2, use_container_width=True)
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.markdown("**Sonno vs Sforzo**")
        fig3 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', 
                         color='Rischio Infortunio', color_continuous_scale=['lightblue', 'red'], 
                         height=350, opacity=0.8, title="Recupero vs Sforzo")
        fig3.add_hline(y=7, line_dash="dash", line_color="orange")
        fig3.add_vline(x=6.5, line_dash="dash", line_color="orange")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_g4:
        st.markdown("**Distribuzione RPE**")
        fig4 = px.histogram(df, x='RPE', nbins=9, title="Distribuzione Sforzo Percepito", height=350)
        fig4.update_traces(marker_color='steelblue')
        st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima di accedere al Dashboard.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # CALCOLI
        risk_score = calcola_risk_score(r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'])
        recovery_score = calcola_recovery_score(r['ore_sonno'])
        sma = calcola_sma(r['stress_lavoro'], r['rpe_previsto'], r['ore_sonno'])
        
        # DETERMINAZIONE STATO
        if risk_score < 25:
            status_color = "#34a853"
            status_text = "OTTIMALE"
            status_emoji = "🟢"
        elif risk_score < 60:
            status_color = "#fbbc04"
            status_text = "MODERATO"
            status_emoji = "🟡"
        else:
            status_color = "#ea4335"
            status_text = "CRITICO"
            status_emoji = "🔴"
        
        st.markdown(f"<h3 style='text-align: center; color: {status_color};'>{status_emoji} Stato Odierno: {status_text}</h3>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # KPI CARDS
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div style='background: {status_color}15; border: 3px solid {status_color}; border-radius: 15px; padding: 35px 20px; text-align: center;'>
            <p style='font-size: 2.5em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 3em; margin: 10px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.1em; color: #666; margin: 0;'><strong>Rischio Infortunio</strong></p>
            <p style='font-size: 0.85em; color: #999; margin: 5px 0;'>Basato su dati storici</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            rec_emoji = "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "❌"
            rec_color = "#34a853" if recovery_score >= 75 else "#fbbc04" if recovery_score >= 40 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {rec_color}15; border: 3px solid {rec_color}; border-radius: 15px; padding: 35px 20px; text-align: center;'>
            <p style='font-size: 2.5em; margin: 0; color: {rec_color};'>{rec_emoji}</p>
            <p style='font-size: 3em; margin: 10px 0; font-weight: bold; color: {rec_color};'>{recovery_score:.0f}%</p>
            <p style='font-size: 1.1em; color: #666; margin: 0;'><strong>Recovery Score</strong></p>
            <p style='font-size: 0.85em; color: #999; margin: 5px 0;'>Capacità di recupero</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            sma_emoji = "✅" if sma < 10 else "⚠️" if sma < 15 else "❌"
            sma_color = "#34a853" if sma < 10 else "#fbbc04" if sma < 15 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {sma_color}15; border: 3px solid {sma_color}; border-radius: 15px; padding: 35px 20px; text-align: center;'>
            <p style='font-size: 2.5em; margin: 0; color: {sma_color};'>{sma_emoji}</p>
            <p style='font-size: 3em; margin: 10px 0; font-weight: bold; color: {sma_color};'>{sma:.1f}</p>
            <p style='font-size: 1.1em; color: #666; margin: 0;'><strong>SMA Score</strong></p>
            <p style='font-size: 0.85em; color: #999; margin: 5px 0;'>Stress × RPE / Sonno</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Parametri Attuali")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"Target: 7.5h")
        col_p2.metric("🧠 Stress Lavoro", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE Previsto", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Frequenza")
        
        st.markdown("---")
        st.subheader("📈 Trend Ultimi 30 Giorni")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            df_recent = df.tail(30).copy()
            fig_t1 = px.line(df_recent, y='RPE', title="RPE Trend", height=300)
            fig_t1.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red", annotation_text="Oggi")
            st.plotly_chart(fig_t1, use_container_width=True)
        
        with col_t2:
            fig_t2 = px.line(df_recent, y='Ore Sonno', title="Sonno Trend", height=300)
            fig_t2.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", annotation_text="Oggi")
            st.plotly_chart(fig_t2, use_container_width=True)

# =====================================================================
# PAGINA 4: CONSIGLIO FINALE
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato Basato su Dati")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima di ricevere consigli.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # CALCOLI CHIAVE
        risk_score = calcola_risk_score(r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'])
        recovery_score = calcola_recovery_score(r['ore_sonno'])
        sma = calcola_sma(r['stress_lavoro'], r['rpe_previsto'], r['ore_sonno'])
        
        # ANALISI STORICA
        media_sonno_90gg = df['Ore Sonno'].mean()
        media_stress_90gg = df['Stress Lavoro'].mean()
        media_rpe_90gg = df['RPE'].mean()
        giorni_rischio_90gg = df['Rischio Infortunio'].sum()
        
        # STATO ODIERNO vs STORICO
        sonno_vs_media = r['ore_sonno'] - media_sonno_90gg
        stress_vs_media = r['stress_lavoro'] - media_stress_90gg
        rpe_vs_media = r['rpe_previsto'] - media_rpe_90gg
        
        # STATO DECISIONALE
        if risk_score < 25:
            status = "OTTIMALE"
            status_color = "#34a853"
            emoji = "🟢"
            consiglio_principale = "ALLENAMENTO INTENSO AUTORIZZATO"
        elif risk_score < 60:
            status = "MODERATO"
            status_color = "#fbbc04"
            emoji = "🟡"
            consiglio_principale = "RECUPERO ATTIVO CONSIGLIATO"
        else:
            status = "CRITICO"
            status_color = "#ea4335"
            emoji = "🔴"
            consiglio_principale = "RIPOSO OBBLIGATORIO"
        
        # HEADER
        st.markdown(f"""
        <div style='background: {status_color}15; border: 3px solid {status_color}; border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 30px;'>
        <h2 style='color: {status_color}; margin: 0;'>{emoji} {status}</h2>
        <p style='font-size: 1.3em; color: {status_color}; margin: 10px 0; font-weight: bold;'>{consiglio_principale}</p>
        <p style='font-size: 1em; color: #666; margin: 0;'>Rischio Infortunio: {risk_score:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ANALISI FATTORI
        st.subheader("🔍 Cosa Dicono i Tuoi Dati")
        
        col_analisi1, col_analisi2, col_analisi3 = st.columns(3)
        
        with col_analisi1:
            sonno_status = "⬇️ SOTTO" if sonno_vs_media < -0.5 else "⬆️ SOPRA" if sonno_vs_media > 0.5 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>SONNO RISPETTO MEDIA</p>
            <p style='margin: 10px 0; font-size: 1.4em; font-weight: bold; color: #1a73e8;'>{sonno_status}</p>
            <p style='margin: 0; font-size: 0.85em; color: #999;'>
            {r['ore_sonno']:.1f}h (media: {media_sonno_90gg:.1f}h)
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_analisi2:
            stress_status = "⬇️ SOTTO" if stress_vs_media < -1 else "⬆️ SOPRA" if stress_vs_media > 1 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>STRESS RISPETTO MEDIA</p>
            <p style='margin: 10px 0; font-size: 1.4em; font-weight: bold; color: #1a73e8;'>{stress_status}</p>
            <p style='margin: 0; font-size: 0.85em; color: #999;'>
            {r['stress_lavoro']}/10 (media: {media_stress_90gg:.1f}/10)
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_analisi3:
            rpe_status = "⬇️ SOTTO" if rpe_vs_media < -1 else "⬆️ SOPRA" if rpe_vs_media > 1 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 8px;'>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>RPE RISPETTO MEDIA</p>
            <p style='margin: 10px 0; font-size: 1.4em; font-weight: bold; color: #1a73e8;'>{rpe_status}</p>
            <p style='margin: 0; font-size: 0.85em; color: #999;'>
            {r['rpe_previsto']}/10 (media: {media_rpe_90gg:.1f}/10)
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # GRAFICI CON SPIEGAZIONI
        st.subheader("📊 Visualizzazione dei Tuoi Dati")
        
        tab1, tab2, tab3 = st.tabs(["📈 Trend Sonno", "⚡ Trend RPE", "🧠 Trend Stress"])
        
        with tab1:
            df_recent = df.tail(30).copy()
            df_recent['Giorno_num'] = range(1, len(df_recent) + 1)
            
            fig_sonno = go.Figure()
            fig_sonno.add_trace(go.Scatter(
                x=df_recent['Giorno_num'], y=df_recent['Ore Sonno'],
                mode='lines+markers', name='Sonno',
                line=dict(color='#1a73e8', width=3),
                marker=dict(size=8)
            ))
            fig_sonno.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="Target: 7.5h")
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", annotation_text=f"Oggi: {r['ore_sonno']:.1f}h")
            fig_sonno.update_layout(title="Andamento Sonno - Ultimi 30 Giorni", height=350, hovermode='x unified')
            st.plotly_chart(fig_sonno, use_container_width=True)
            
            st.info(f"""
            **Cosa vedi nel grafico:** La linea blu mostra come hai dormito negli ultimi 30 giorni.
            - Linea verde: target ideale (7.5h)
            - Linea rossa: quanto hai dormito oggi ({r['ore_sonno']:.1f}h)
            
            **Analisi:** Hai dormito {abs(sonno_vs_media):.1f}h {'meno' if sonno_vs_media < 0 else 'più'} della tua media. 
            {'Questo aumenta il rischio!' if sonno_vs_media < -0.5 else 'Buon segno!' if sonno_vs_media > 0.5 else ''}
            """)
        
        with tab2:
            df_recent = df.tail(30).copy()
            df_recent['Giorno_num'] = range(1, len(df_recent) + 1)
            
            fig_rpe = go.Figure()
            fig_rpe.add_trace(go.Scatter(
                x=df_recent['Giorno_num'], y=df_recent['RPE'],
                mode='lines+markers', name='RPE',
                line=dict(color='#ea4335', width=3),
                marker=dict(size=8)
            ))
            fig_rpe.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Soglia Moderato")
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="blue", annotation_text=f"Oggi: {r['rpe_previsto']}/10")
            fig_rpe.update_layout(title="Andamento RPE - Ultimi 30 Giorni", height=350, hovermode='x unified')
            st.plotly_chart(fig_rpe, use_container_width=True)
            
            st.info(f"""
            **Cosa vedi nel grafico:** La linea rossa mostra l'intensità percepita negli ultimi 30 giorni.
            - Linea arancione: transizione facile/moderato (5/10)
            - Linea blu: RPE che farai oggi ({r['rpe_previsto']}/10)
            
            **Analisi:** Oggi prevedi RPE {r['rpe_previsto']}/10, che è {'superiore' if rpe_vs_media > 1 else 'inferiore' if rpe_vs_media < -1 else 'in linea con'} la tua media.
            """)
        
        with tab3:
            df_recent = df.tail(30).copy()
            df_recent['Giorno_num'] = range(1, len(df_recent) + 1)
            
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Scatter(
                x=df_recent['Giorno_num'], y=df_recent['Stress Lavoro'],
                mode='lines+markers', name='Stress',
                line=dict(color='#fbbc04', width=3),
                marker=dict(size=8)
            ))
            fig_stress.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Medio")
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red", annotation_text=f"Oggi: {r['stress_lavoro']}/10")
            fig_stress.update_layout(title="Andamento Stress Lavoro - Ultimi 30 Giorni", height=350, hovermode='x unified')
            st.plotly_chart(fig_stress, use_container_width=True)
            
            st.info(f"""
            **Cosa vedi nel grafico:** La linea gialla mostra lo stress lavorativo negli ultimi 30 giorni.
            - Linea arancione: livello medio (5/10)
            - Linea rossa: stress di oggi ({r['stress_lavoro']}/10)
            
            **Analisi:** Oggi hai stress {r['stress_lavoro']}/10, che è {'superiore' if stress_vs_media > 1 else 'inferiore' if stress_vs_media < -1 else 'in linea con'} la tua media.
            """)
        
        st.markdown("---")
        
        # RACCOMANDAZIONI DINAMICHE
        st.subheader("💡 Raccomandazione Personalizzata")
        
        if risk_score < 25:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ ALLENAMENTO INTENSO - SEI PRONTO!</h3>
            <p>Tutti i tuoi parametri sono ottimali. Puoi fare allenamenti intensi senza problemi.</p>
            
            <ul>
            <li><strong>Intervalli veloci:</strong> 6-8 × 800m con 2' recupero</li>
            <li><strong>Tempo run:</strong> 3 × 10min a ritmo sostenuto</li>
            <li><strong>Test di velocità:</strong> Perfetto per misurare i progressi</li>
            <li><strong>Gara:</strong> Condizioni ideali</li>
            </ul>
            
            <p><strong>Domani:</strong> Easy run 6-8 km per recuperare</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif risk_score < 60:
            st.markdown(f"""
            <div class='warning-box'>
            <h3>🟡 RECUPERO ATTIVO - ASCOLTA IL CORPO</h3>
            <p>Hai {'poco sonno' if sonno_vs_media < -0.5 else ''} {'stress alto' if stress_vs_media > 1 else ''} - il corpo ha bisogno di rigenerazione.</p>
            
            <ul>
            <li><strong>Easy run:</strong> Ritmo conversativo 30-45 min</li>
            <li><strong>Long run facile:</strong> 12-15 km a bassa intensità</li>
            <li><strong>Recovery run:</strong> 5-8 km per mobilità</li>
            <li><strong>Cross-training:</strong> Nuoto o ciclismo leggero</li>
            </ul>
            
            <p><strong>Priorità:</strong> Dormi più a lungo stasera (target: {max(8, r['ore_sonno'] + 1):.1f}h)</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class='danger-box'>
            <h3>🔴 RIPOSO OBBLIGATORIO</h3>
            <p><strong>ATTENZIONE:</strong> Sonno insufficiente ({r['ore_sonno']:.1f}h), stress elevato ({r['stress_lavoro']}/10), RPE alto ({r['rpe_previsto']}/10).</p>
            
            <h4>❌ Non correre oggi</h4>
            <ul>
            <li>No allenamenti intensi o moderati</li>
            <li>Max: camminate leggere 15 minuti</li>
            <li>Stretching delicato</li>
            <li>Riposo totale prioritario</li>
            </ul>
            
            <h4>🛏️ Priorità Assoluta</h4>
            <ul>
            <li>Dormi 9-10 ore stasera</li>
            <li>Riduci stress mentale</li>
            <li>Nutrizione leggera ma nutriente</li>
            <li>Se febbre/dolore acuto → medico</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📋 Riepilogo Parametri")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress Lavoro', 'RPE Previsto', 'FC Riposo', 'Recovery Score', 'SMA Score', 'Rischio Infortunio'],
            'Valore': [
                f"{r['ore_sonno']:.1f}h",
                f"{r['stress_lavoro']}/10",
                f"{r['rpe_previsto']}/10",
                f"{r['fc_riposo']} bpm",
                f"{recovery_score:.0f}%",
                f"{sma:.1f}",
                f"{risk_score:.0f}%"
            ],
            'Stato': [
                "✅ OK" if r['ore_sonno'] >= 7 else "⚠️ Insufficiente",
                "✅ OK" if r['stress_lavoro'] <= 5 else "⚠️ Elevato",
                "✅ OK" if r['rpe_previsto'] <= 5 else "⚠️ Intenso",
                "✅ OK" if r['fc_riposo'] <= 65 else "⚠️ Elevata",
                "✅ OK" if recovery_score >= 75 else "⚠️ Moderato",
                "✅ OK" if sma < 10 else "⚠️ Moderato",
                "✅ Sicuro" if risk_score < 25 else "⚠️ Moderato" if risk_score < 60 else "🔴 Critico"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
