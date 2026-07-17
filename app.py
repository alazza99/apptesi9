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

# --- UI/UX UPGRADE: Tema Dark Stile Whoop ---
st.markdown("""
<style>
    /* Sfondo globale scuro e font moderno */
    .stApp { background-color: #0b0f19; color: #f8f9fa; font-family: 'Inter', 'Segoe UI', sans-serif; }
    
    /* Stile Headers */
    h1 { color: #ffffff; text-align: center; margin-bottom: 30px; font-size: 2.5em; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #ffffff; padding-bottom: 15px; margin-bottom: 20px; font-size: 1.8em; font-weight: 700; border-bottom: 1px solid #1f2937; }
    h3 { color: #e5e7eb; font-size: 1.3em; font-weight: 600; }
    
    /* Box Informativi Modernizzati (Glassmorphism effect) */
    .info-box { background: rgba(26, 115, 232, 0.1); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .success-box { background: rgba(52, 168, 83, 0.1); border-left: 4px solid #10b981; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .warning-box { background: rgba(251, 188, 4, 0.1); border-left: 4px solid #f59e0b; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .danger-box { background: rgba(234, 67, 53, 0.1); border-left: 4px solid #ef4444; padding: 20px; border-radius: 12px; margin: 20px 0; color: #d1d5db; border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05); }

    /* Custom KPI Cards */
    .kpi-card { background: #111827; border-radius: 16px; padding: 30px 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.3); border: 1px solid #1f2937; transition: transform 0.2s; }
    .kpi-card:hover { transform: translateY(-5px); }
</style>
""", unsafe_allow_html=True)

# Imposta il tema Plotly Dark di default per integrarsi col nuovo sfondo
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
            obj_lt = st.text_input("Obiettivo Lungo Termine", placeholder="Es: Maratona < 3:30")
        
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
        st.success("✓ Analisi completata e caricata nel modello!")

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["📏 Volume", "❤️ Intensità", "😴 Recupero", "📋 Tabella"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**KM per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W')).agg({
                'Distanza (km)': 'sum',
                'RPE': 'mean'
            }).reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', 
                         height=350, color='Distanza (km)', color_continuous_scale='Blues')
            fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("Incremento graduale ideale: 10% settimanale. Evita picchi improvvisi.")
        
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=350, markers=True)
            fig_cum.update_traces(line_color="#3b82f6", marker_color="#60a5fa")
            fig_cum.update_layout(xaxis_title="Data", yaxis_title="KM Cumulativi", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cum, use_container_width=True)
            
            st.info(f"Total: {df['Cumulativa'].iloc[-1]:.0f} km in 90 giorni = {df['Cumulativa'].iloc[-1]/90:.1f} km/giorno")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                            size='Distanza (km)', color='RPE',
                            color_continuous_scale='Viridis', height=350, opacity=0.8)
            fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info("Punti bassi = cuore più efficiente. Colore = RPE.")
        
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=350, color_discrete_sequence=['#3b82f6'])
            fig3.update_layout(xaxis_title="RPE (1-10)", yaxis_title="Giorni", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            
            easy_pct = (df['RPE'] <= 3).sum() / len(df) * 100
            hard_pct = (df['RPE'] >= 7).sum() / len(df) * 100
            
            fig3.add_vline(x=3.5, line_dash="dash", line_color="#10b981")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="#ef4444")
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info(f"{easy_pct:.0f}% Easy + {hard_pct:.0f}% Intenso = Polarized Training")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=350, markers=True)
            fig_sleep.update_traces(line_color="#8b5cf6", marker_color="#a78bfa")
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="#10b981")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="#ef4444")
            fig_sleep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sleep, use_container_width=True)
            
            media_sonno = df['Ore Sonno'].mean()
            st.info(f"Media: {media_sonno:.1f}h/notte | Target: 7.5h")
        
        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', 
                            size='Distanza (km)', color='Rischio Infortunio',
                            color_continuous_scale=['#3b82f6', '#ef4444'],
                            height=350, opacity=0.9)
            fig4.add_hline(y=7, line_dash="dash", line_color="#f59e0b")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="#f59e0b")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
            
            giorni_rischio = df['Rischio Infortunio'].sum()
            st.info(f"Giorni a rischio: {giorni_rischio} su {len(df)} ({giorni_rischio/len(df)*100:.1f}%)")
    
    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: '🔴' if x == 1 else '✅')
        st.dataframe(tab_data, use_container_width=True, hide_index=True)

elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima.")
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
        
        if risk_score < 25:
            status_color = "#10b981" # Green
            status_text = "OTTIMALE"
        elif risk_score < 60:
            status_color = "#f59e0b" # Yellow
            status_text = "MODERATO"
        else:
            status_color = "#ef4444" # Red
            status_text = "CRITICO"
        
        st.markdown(f"<h3 style='text-align: center; color: {status_color}; font-size: 2em; letter-spacing: 2px;'>{status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 4px solid {status_color};'>
                <div class='kpi-title'>Rischio Infortunio</div>
                <div class='kpi-value' style='color: {status_color};'>{risk_score:.0f}%</div>
                <hr style='border-color: #374151; margin: 15px 0;'>
                <p style='font-size: 0.85em; color: #9ca3af; margin: 0;'>
                <strong>0-25%:</strong> Sicuro 🟢<br>
                <strong>25-60%:</strong> Moderato 🟡<br>
                <strong>60-100%:</strong> Riposo 🔴
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            rec_color = "#10b981" if recovery_score >= 75 else "#f59e0b" if recovery_score >= 40 else "#ef4444"
            
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 4px solid {rec_color};'>
                <div class='kpi-title'>Recovery Score</div>
                <div class='kpi-value' style='color: {rec_color};'>{recovery_score:.0f}%</div>
                <hr style='border-color: #374151; margin: 15px 0;'>
                <p style='font-size: 0.85em; color: #9ca3af; margin: 0;'>
                Target sonno: <strong>7.5h</strong><br>
                per massimizzare la<br>
                capacità di recupero.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            sma_color = "#10b981" if sma < 10 else "#f59e0b" if sma < 15 else "#ef4444"
            
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 4px solid {sma_color};'>
                <div class='kpi-title'>SMA Score</div>
                <div class='kpi-value' style='color: {sma_color};'>{sma:.1f}</div>
                <hr style='border-color: #374151; margin: 15px 0;'>
                <p style='font-size: 0.85em; color: #9ca3af; margin: 0;'>
                Equilibrio globale tra<br>
                stress, sforzo fisico<br>
                e recupero.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📊 Visualizzazioni Avanzate")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Level", 'font': {'color': '#9ca3af'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': status_color, 'thickness': 0.75},
                    'bgcolor': "#1f2937",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.1)"},
                        {'range': [25, 60], 'color': "rgba(245, 158, 11, 0.1)"},
                        {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                    ]
                },
                number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}}
            ))
            fig_gauge.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno\n(h)', 'Stress\n(1-10)', 'RPE\n(1-10)', 'Recovery\n(%)'],
                fill='toself',
                name='Parametri',
                marker=dict(color=status_color),
                line=dict(color=status_color)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], gridcolor='#374151'),
                    angularaxis=dict(gridcolor='#374151')
                ),
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"vs 7.5h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Base")

elif pagina == "🔮 ML Explained":
    st.title("🔮 AI/Machine Learning - Explainability")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cos'è il Machine Learning?</h3>
    <p style='color: #d1d5db;'>L'algoritmo impara dai dati storici per fare previsioni su nuovi dati. Analizzando i 90 giorni passati, identifica pattern complessi che portano al rischio di infortunio e calcola la tua probabilità odierna.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        df = st.session_state.dati.copy()
        num_giorni = len(df)
        num_rischi = df['Rischio Infortunio'].sum()
        
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
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎓 Spiegazione", "📊 Feature Importance", "🔬 Confusion Matrix", "📈 Metriche", "🧮 Calcolo"])
        
        with tab1:
            st.markdown(f"""
            <div class='kpi-card' style='text-align: left; background-color: #1f2937;'>
            <h3 style='color: white;'>🌳 Random Forest Architecture</h3>
            <p><strong>Dataset Storico:</strong> {num_giorni} giorni | {num_rischi} giorni a rischio ({num_rischi/num_giorni*100:.1f}%)</p>
            <p><strong>Come opera:</strong> Genera 100 alberi decisionali indipendenti. Ognuno valuta i parametri e "vota". La probabilità finale è la media di questi voti.</p>
            <pre style='background: #111827; padding: 15px; border-radius: 8px; font-size: 0.85em; color: #10b981; border: 1px solid #374151;'>
ALBERO 1:
├─ IF Ore_Sonno < 6.5
│  ├─ IF RPE > 7 → RISCHIO ALTO ✓
│  └─ ELSE → MODERATO

ALBERO 2:
├─ IF Stress > 7
│  ├─ IF Sonno < 6 → RISCHIO ✓
│  └─ ELSE → MODERATO
            </pre>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in imp_data]
            importances_sorted = [x[1] for x in imp_data]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=features_sorted,
                x=[x*100 for x in importances_sorted],
                orientation='h',
                marker=dict(color=importances_sorted, colorscale='Reds', showscale=False),
                text=[f'{x*100:.1f}%' for x in importances_sorted],
                textposition='auto',
                textfont=dict(color='white')
            ))
            fig_imp.update_layout(
                title="Pesi delle Feature nel Modello",
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor='#374151')
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with tab3:
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predetto: Sicuro', 'Predetto: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 24, "color": "white"},
                colorscale='RdYlGn_r'
            ))
            fig_cm.update_layout(
                title="Matrice di Confusione - Accuratezza Dati Storici",
                height=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with tab4:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Esattezza 'Rischio'")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Rischi individuati")
            
        with tab5:
            if st.session_state.analisi_fatta:
                r = st.session_state.risultati_analisi
                input_data = np.array([[5, r['ore_sonno'], r['stress_lavoro'], 
                                       100 + r['rpe_previsto']*10, r['rpe_previsto']]])
                input_scaled = scaler.transform(input_data)
                
                prob_rischio = rf_model.predict_proba(input_scaled)[0][1] * 100
                
                color_ml = "#ef4444" if prob_rischio >= 60 else "#f59e0b" if prob_rischio >= 25 else "#10b981"
                
                st.markdown(f"""
                <div class='kpi-card' style='border-top: 4px solid {color_ml};'>
                    <h3 style='color: white; margin:0;'>Probabilità Predetta AI</h3>
                    <p style='font-size: 4em; font-weight: 800; color: {color_ml}; margin: 10px 0;'>{prob_rischio:.1f}%</p>
                    <p style='color: #9ca3af;'>Probabilità di infortunio calcolata su tutti i 100 alberi decisionali del cluster.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ Completa il questionario per vedere il calcolo personalizzato.")
                
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

elif pagina == "💡 Consiglio Finale":
    st.title("💡 Piano di Azione Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima.")
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
        
        media_sonno_90 = df['Ore Sonno'].mean()
        media_stress_90 = df['Stress Lavoro'].mean()
        media_rpe_90 = df['RPE'].mean()
        
        sonno_vs_media = r['ore_sonno'] - media_sonno_90
        stress_vs_media = r['stress_lavoro'] - media_stress_90
        rpe_vs_media = r['rpe_previsto'] - media_rpe_90
        
        if risk_score < 25:
            title = "ALLENAMENTO INTENSO AUTORIZZATO"
            color = "#10b981"
            box_class = "success-box"
        elif risk_score < 60:
            title = "RECUPERO ATTIVO CONSIGLIATO"
            color = "#f59e0b"
            box_class = "warning-box"
        else:
            title = "RIPOSO OBBLIGATORIO"
            color = "#ef4444"
            box_class = "danger-box"
        
        st.markdown(f"""
        <div class='kpi-card' style='border: 2px solid {color}; background-color: rgba(0,0,0,0.5);'>
            <h2 style='color: {color}; margin: 0; border: none;'>{title}</h2>
            <p style='font-size: 1.1em; color: #d1d5db; margin-top: 10px;'>Rischio Infortunio Globale: <strong style='color: white;'>{risk_score:.0f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Metriche vs Baseline Storica")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            sonno_badge = "SOTTO MEDIA" if sonno_vs_media < -0.5 else "SOPRA MEDIA" if sonno_vs_media > 0.5 else "IN MEDIA"
            s_color = "#ef4444" if sonno_vs_media < -0.5 else "#10b981" if sonno_vs_media > 0.5 else "#9ca3af"
            st.markdown(f"""
            <div class='kpi-card' style='padding: 20px;'>
            <p style='margin: 0; font-size: 0.9em; font-weight: bold; color: {s_color};'>{sonno_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold; color: white;'>{r['ore_sonno']:.1f}h</p>
            <p style='margin: 0; font-size: 0.8em; color: #6b7280;'>vs Storico {media_sonno_90:.1f}h</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_a2:
            stress_badge = "SOTTO MEDIA" if stress_vs_media < -1 else "SOPRA MEDIA" if stress_vs_media > 1 else "IN MEDIA"
            st_color = "#10b981" if stress_vs_media < -1 else "#ef4444" if stress_vs_media > 1 else "#9ca3af"
            st.markdown(f"""
            <div class='kpi-card' style='padding: 20px;'>
            <p style='margin: 0; font-size: 0.9em; font-weight: bold; color: {st_color};'>{stress_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold; color: white;'>{r['stress_lavoro']}/10</p>
            <p style='margin: 0; font-size: 0.8em; color: #6b7280;'>vs Storico {media_stress_90:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_a3:
            rpe_badge = "SOTTO MEDIA" if rpe_vs_media < -1 else "SOPRA MEDIA" if rpe_vs_media > 1 else "IN MEDIA"
            rp_color = "#9ca3af"
            st.markdown(f"""
            <div class='kpi-card' style='padding: 20px;'>
            <p style='margin: 0; font-size: 0.9em; font-weight: bold; color: {rp_color};'>{rpe_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold; color: white;'>{r['rpe_previsto']}/10</p>
            <p style='margin: 0; font-size: 0.8em; color: #6b7280;'>vs Storico {media_rpe_90:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if risk_score < 25:
                st.markdown("""
                <div class='success-box'>
                <h3 style='color: #10b981;'>⚡ Prestazione Ottimale</h3>
                <ul style='color: #d1d5db; line-height: 1.6;'>
                <li><strong>Intervalli Veloci:</strong> 6-8 × 800m (2' recupero)</li>
                <li><strong>Tempo Run:</strong> 3 × 10min a ritmo sostenuto</li>
                <li><strong>Ripetute:</strong> 5 × 2km (3' recupero)</li>
                <li><strong>Test Velocità:</strong> Perfetto per misurare progressi</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score < 60:
                st.markdown("""
                <div class='warning-box'>
                <h3 style='color: #f59e0b;'>🟡 Moderare il Carico</h3>
                <ul style='color: #d1d5db; line-height: 1.6;'>
                <li><strong>Easy Run:</strong> Ritmo conversativo 30-40 min</li>
                <li><strong>Recovery Run:</strong> 5-8 km per mobilità</li>
                <li><strong>Cross-Training:</strong> Nuoto/ciclismo leggero 30-45 min</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='danger-box'>
                <h3 style='color: #ef4444;'>🔴 Stop Precauzionale</h3>
                <ul style='color: #d1d5db; line-height: 1.6;'>
                <li>No allenamenti intensi o moderati</li>
                <li>Max concesso: camminate 15 min</li>
                <li>Priorità assoluta a sonno e reidratazione</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
        with col_rec2:
            st.markdown("""
            <div class='info-box'>
            <h3 style='color: #3b82f6;'>🔄 Routine Consigliata</h3>
            <ul style='color: #d1d5db; line-height: 1.6;'>
            <li><strong>Entro 30 min:</strong> Window metabolica (Proteine + Carbs)</li>
            <li><strong>Idratazione:</strong> Target +500ml per ogni ora di sforzo</li>
            <li><strong>Sleep Hygiene:</strong> Anticita di 30 min l'orario a letto stasera per compensare i parametri.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
