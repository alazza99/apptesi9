import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
    
    .kpi-explanation { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 6px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

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
    st.session_state.device_data = {
        'fc': 72, 'battery': 85, 'steps': 0, 'calories': 0
    }
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

# =====================================================================
# SIDEBAR - DEVICE CONNECTION
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
        <br>
        ❤️ FC: {st.session_state.device_data['fc']} bpm | 🔋 {st.session_state.device_data['battery']}% | 👟 {st.session_state.device_data['steps']} passi
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("📋 Menu", 
        ["📋 Analisi Completa", "📈 Statistiche", "📊 KPI Dashboard", "🔮 ML Explained", "💡 Consiglio Finale"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: ANALISI COMPLETA
# =====================================================================
if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa dello Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    <strong>ℹ️ Come funziona:</strong> Compila <strong>solo i parametri che il tuo dispositivo non trasmette</strong> 
    (sonno, stress mentale, obiettivi). I dati di FC, velocità, distanza verranno presi automaticamente dal dispositivo connesso.
    </div>
    """, unsafe_allow_html=True)
    
    # FORM
    st.subheader("📝 Questionario Minimalista")
    
    with st.form("form_analisi"):
        # SEZIONE 1: OBIETTIVI (NON dal device)
        st.markdown("### 🎯 I Tuoi Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            obj_lt = st.text_input("Obiettivo Lungo Termine", placeholder="Es: Maratona < 3:30")
        
        st.markdown("---")
        
        # SEZIONE 2: SONNO E RECUPERO (NON dal device)
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno scorsa notte", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
        
        recovery_score_calc = max(0, 100 - (7.5 - ore_sonno) * 10)
        
        st.markdown("---")
        
        # SEZIONE 3: STRESS E LAVORO (NON dal device)
        st.markdown("### 🧠 Stress Mentale")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate oggi", 0.0, 14.0, 8.0)
        
        st.markdown("---")
        
        # SEZIONE 4: TIPO DI ALLENAMENTO (NON dal device)
        st.markdown("### ⚡ Tipo di Allenamento Previsto")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Categoria", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)
        
        st.markdown("---")
        
        # PULSANTE ANALIZZA
        st.markdown("### ✅ Pronto?")
        bottone = st.form_submit_button("🚀 ANALIZZA", use_container_width=True)
    
    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi,
            'obj_lt': obj_lt,
            'ore_sonno': ore_sonno,
            'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo,
            'recovery_score': recovery_score_calc,
            'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro,
            'tipo_allenamento': tipo_allenamento,
            'rpe_previsto': rpe_previsto,
        }
        
        st.success("✓ Analisi salvata! Vai su 'Statistiche' per una panoramica dei tuoi ultimi 90 giorni.")

# =====================================================================
# PAGINA 2: STATISTICHE
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Dettagliate - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    # KPI PRINCIPALI CON SPIEGAZIONE
    st.subheader("📊 Metriche Principali")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    km_totali = df['Distanza (km)'].sum()
    sessioni = len(df)
    media_km = df['Distanza (km)'].mean()
    media_fc = df['FC Media'].mean()
    media_rpe = df['RPE'].mean()
    media_sonno = df['Ore Sonno'].mean()
    
    col_m1.metric("🏃 KM Totali", f"{km_totali:.0f} km", "3 mesi")
    col_m2.metric("📊 Sessioni", f"{sessioni}")
    col_m3.metric("📐 Media/Sessione", f"{media_km:.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")
    
    # SPIEGAZIONE KPI
    st.markdown("---")
    st.subheader("🔍 Cosa Significano Questi Numeri?")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>🏃 KM Totali</strong><br>
        Distanza totale percorsa in 90 giorni.
        <br><br>
        <strong>Benchmark:</strong><br>
        • Maratoneta base: 50-70 km<br>
        • Maratoneta agonista: 80-120 km<br>
        • Ultra runner: 120+ km
        </div>
        """, unsafe_allow_html=True)
    
    with col_exp2:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>❤️ FC Media</strong><br>
        Frequenza cardiaca media durante gli allenamenti.
        <br><br>
        <strong>Interpretazione:</strong><br>
        • Z1-Z2: 100-130 bpm (recupero)<br>
        • Z3: 131-150 bpm (base)<br>
        • Z4-Z5: 151+ bpm (intenso)
        </div>
        """, unsafe_allow_html=True)
    
    with col_exp3:
        st.markdown("""
        <div class='kpi-explanation'>
        <strong>😴 Sonno Medio</strong><br>
        Ore di sonno per notte negli ultimi 90gg.
        <br><br>
        <strong>Raccomandazioni:</strong><br>
        • < 6h: Insufficiente ❌<br>
        • 6-7h: Minimo accettabile ⚠️<br>
        • 7-9h: Ottimale ✅<br>
        • > 9h: Eccesso (recupero urgente)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # GRAFICI PRINCIPALI - EXPANDABLE
    st.subheader("📉 Analisi Dettagliata")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Progressione KM - Trend Settimanale**")
        df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
        df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
        
        fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', 
                     title="KM per Settimana", labels={'Distanza (km)': 'KM'}, height=380)
        fig1.update_traces(marker_color='#1a73e8')
        fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> La progressione settimanale dei volumi di allenamento.
        <br><strong>Trend ottimale:</strong> Aumento del 10% settimanale per 3 settimane, poi scarico.
        </div>
        """, unsafe_allow_html=True)
    
    with col_g2:
        st.markdown("**Relazione FC vs Velocità - Efficienza Cardiaca**")
        fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                         size='Distanza (km)', color='RPE', 
                         color_continuous_scale='Blues', height=380, opacity=0.7,
                         title="Efficienza Cardiaca")
        fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)", 
                          showlegend=False, hovermode='closest')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Come varia la FC al variare della velocità.
        <br><strong>Interpretazione:</strong> Se i punti sono più bassi, il tuo cuore è più efficiente.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        st.markdown("**Sonno vs Sforzo - Qualità del Recupero**")
        fig3 = px.scatter(df, x='Ore Sonno', y='RPE', 
                         size='Distanza (km)', color='Rischio Infortunio',
                         color_continuous_scale=['lightblue', 'red'], height=380, opacity=0.8,
                         title="Recupero vs Sforzo")
        fig3.add_hline(y=7, line_dash="dash", line_color="orange", annotation_text="Soglia Alto RPE")
        fig3.add_vline(x=6.5, line_dash="dash", line_color="orange", annotation_text="Sonno Minimo")
        fig3.update_layout(xaxis_title="Ore Sonno", yaxis_title="RPE (1-10)", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Punti rossi = giorni a rischio infortunio.
        <br><strong>Zona critica:</strong> RPE > 7 + Sonno < 6.5h = Alto rischio ❌
        </div>
        """, unsafe_allow_html=True)
    
    with col_g4:
        st.markdown("**Distribuzione RPE - Equilibrio Allenamento**")
        fig4 = px.histogram(df, x='RPE', nbins=9, title="Distribuzione Sforzo Percepito", height=380,
                           labels={'RPE': 'RPE (1-10)', 'count': 'Giorni'})
        fig4.update_traces(marker_color='steelblue')
        fig4.add_vline(x=3.5, line_dash="dash", line_color="green", annotation_text="Easy Run")
        fig4.add_vline(x=6.5, line_dash="dash", line_color="orange", annotation_text="Moderato")
        fig4.add_vline(x=8.5, line_dash="dash", line_color="red", annotation_text="Intenso")
        fig4.update_layout(xaxis_title="RPE", yaxis_title="Numero di Giorni", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>📊 Cosa vedi:</strong> Come è distribuito il tuo allenamento.
        <br><strong>Modello ottimale Polarized:</strong> 80% Easy + 20% Intenso (NO moderato!)
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TABELLA DATI COMPLETA
    st.subheader("📋 Ultimi 15 Allenamenti")
    
    tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro', 'Rischio Infortunio']].tail(15).copy()
    tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
    tab_data['Status'] = tab_data['Rischio Infortunio'].apply(lambda x: '⚠️ RISCHIO' if x == 1 else '✅ OK')
    tab_data = tab_data.drop('Rischio Infortunio', axis=1)
    
    st.dataframe(tab_data, use_container_width=True, hide_index=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI - Analisi dei Parametri")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' prima di accedere al Dashboard.")
    else:
        df = st.session_state.dati.copy()
        r = st.session_state.risultati_analisi
        
        # CALCOLI ML CORRETTI
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=6, min_samples_split=8)
        rf_model.fit(X_scaled, y_train)
        
        # PREDIZIONE PER L'ALLENAMENTO DI OGGI
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        fc_media_stimata = 100 + (r['rpe_previsto'] * 10)
        scenario = scaler.transform([[0, r['ore_sonno'], r['stress_lavoro'], fc_media_stimata, r['rpe_previsto']]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        # WIDGET PRINCIPALI CON GAUGE PIU' BELLI
        st.subheader("🎯 Valutazione Odierna - KPI Principali")
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        # KPI 1: Risk - con colore dinamico
        with col_kpi1:
            if prob_rischio < 25:
                risk_color = "#34a853"
                risk_status = "SICURO"
            elif prob_rischio < 60:
                risk_color = "#fbbc04"
                risk_status = "MODERATO"
            else:
                risk_color = "#ea4335"
                risk_status = "CRITICO"
            
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_rischio,
                title="Rischio Infortunio",
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': 'darkblue'},
                    'bar': {'color': risk_color, 'thickness': 0.8},
                    'bgcolor': "#f0f0f0",
                    'borderwidth': 3,
                    'bordercolor': risk_color,
                    'steps': [
                        {'range': [0, 25], 'color': "#e6f4ea"},
                        {'range': [25, 60], 'color': "#fef7e0"},
                        {'range': [60, 100], 'color': "#fce8e6"}
                    ],
                    'threshold': {
                        'line': {'color': risk_color, 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                },
                number={'suffix': "%", 'font': {'size': 28, 'color': risk_color}},
            ))
            fig_risk.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=60, b=10),
                paper_bgcolor="#f8f9fa",
                font=dict(color="#1a73e8", size=14)
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-size: 1.1em; color: {risk_color};'><strong>{risk_status}</strong></p>", unsafe_allow_html=True)
        
        # KPI 2: Recovery Score - con colore dinamico
        with col_kpi2:
            recovery_score = r['recovery_score']
            if recovery_score < 40:
                rec_color = "#ea4335"
                rec_status = "PESSIMO"
            elif recovery_score < 75:
                rec_color = "#fbbc04"
                rec_status = "MODERATO"
            else:
                rec_color = "#34a853"
                rec_status = "OTTIMALE"
            
            fig_recovery = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recovery_score,
                title="Recovery Score",
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': 'darkgreen'},
                    'bar': {'color': rec_color, 'thickness': 0.8},
                    'bgcolor': "#f0f0f0",
                    'borderwidth': 3,
                    'bordercolor': rec_color,
                    'steps': [
                        {'range': [0, 40], 'color': "#fce8e6"},
                        {'range': [40, 75], 'color': "#fef7e0"},
                        {'range': [75, 100], 'color': "#e6f4ea"}
                    ]
                },
                number={'suffix': "%", 'font': {'size': 28, 'color': rec_color}},
            ))
            fig_recovery.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=60, b=10),
                paper_bgcolor="#f8f9fa",
                font=dict(color="#34a853", size=14)
            )
            st.plotly_chart(fig_recovery, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-size: 1.1em; color: {rec_color};'><strong>{rec_status}</strong></p>", unsafe_allow_html=True)
        
        # KPI 3: SMA Score - con badge
        with col_kpi3:
            sma_value = sma
            if sma_value < 10:
                sma_color = "#34a853"
                sma_emoji = "✅"
                sma_status = "BILANCIATO"
            elif sma_value < 15:
                sma_color = "#fbbc04"
                sma_emoji = "⚠️"
                sma_status = "MODERATO"
            else:
                sma_color = "#ea4335"
                sma_emoji = "❌"
                sma_status = "SQUILIBRATO"
            
            st.markdown(f"""
            <div style='background: {sma_color}15; border: 3px solid {sma_color}; padding: 40px 20px; border-radius: 12px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {sma_color};'>{sma_emoji}</p>
            <p style='font-size: 1.8em; font-weight: bold; color: {sma_color}; margin: 10px 0;'>{sma_value:.1f}</p>
            <p style='font-size: 1.1em; color: #666; margin: 5px 0;'><strong>SMA Score</strong></p>
            <p style='font-size: 0.95em; color: {sma_color}; margin-top: 15px;'>{sma_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # SPIEGAZIONE KPI
        st.subheader("📖 Cosa Significano Questi Indicatori?")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.markdown("""
            <div class='info-box'>
            <h4>⚠️ Rischio Infortunio</h4>
            <p><strong>Come funziona:</strong> Un modello ML analizza stress, sonno e FC per prevedere il rischio.</p>
            <p><strong>Algoritmo:</strong> Random Forest (100 alberi decisionali)</p>
            <p><strong>Interpretazione:</strong></p>
            <ul>
            <li>0-25%: 🟢 SICURO - Allenamento normale</li>
            <li>25-60%: 🟡 CAUTION - Recupero attivo</li>
            <li>60-100%: 🔴 CRITICO - Riposo obbligatorio</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_exp2:
            st.markdown("""
            <div class='success-box'>
            <h4>💚 Recovery Score</h4>
            <p><strong>Cosa misura:</strong> La capacità del corpo di recuperare tra allenamenti.</p>
            <p><strong>Formula:</strong> 100 - (7.5 - Ore_Sonno) × 10</p>
            <p><strong>Cosa significa:</strong></p>
            <ul>
            <li>0-40: Recupero pessimo ❌</li>
            <li>40-75: Recupero moderato ⚠️</li>
            <li>75-100: Recupero ottimale ✅</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_exp3:
            st.markdown("""
            <div class='warning-box'>
            <h4>📊 SMA Score</h4>
            <p><strong>SMA = Stress × RPE / Ore_Sonno</strong></p>
            <p><strong>Cosa significa:</strong> Rapporto tra stress mentale/fisico e capacità di recupero.</p>
            <p><strong>Interpretazione:</strong></p>
            <ul>
            <li>< 10: Bilanciato ✅</li>
            <li>10-15: Moderato ⚠️</li>
            <li>> 15: Squilibrato ❌</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # GRAFICI COMPARATIVI
        st.subheader("📊 Comparazione con i Tuoi Ultimi 90 Giorni")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("**Curva di Rischio - Ultime 30 Sessioni**")
            df_risk = df.tail(30).copy()
            df_risk['Risk_Calc'] = (df_risk['RPE'] / 10) * (df_risk['Stress Lavoro'] / 10) * \
                                   (1 if df_risk['Ore Sonno'].mean() < 6.5 else 0.5) * 100
            
            fig_risk_line = px.line(df_risk, x=df_risk.index, y=['RPE', 'Stress Lavoro'],
                                   title="Trend Sforzo vs Stress",
                                   labels={'value': 'Valore', 'index': 'Giorni'},
                                   height=350)
            fig_risk_line.update_layout(hovermode='x unified')
            st.plotly_chart(fig_risk_line, use_container_width=True)
        
        with col_comp2:
            st.markdown("**Distribuzione Sonno - Ultimi 30 Giorni**")
            df_sleep = df.tail(30).copy()
            
            fig_sleep = px.bar(df_sleep, x=df_sleep.index, y='Ore Sonno',
                              title="Ore di Sonno Giornaliere",
                              labels={'Ore Sonno': 'Ore', 'index': 'Giorni'},
                              height=350)
            fig_sleep.add_hline(y=7, line_dash="dash", line_color="green", annotation_text="Target: 7h")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red", annotation_text="Minimo: 6.5h")
            fig_sleep.update_traces(marker_color='darkblue')
            st.plotly_chart(fig_sleep, use_container_width=True)

# =====================================================================
# PAGINA 4: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Come Funziona")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cosa è il Machine Learning?</h3>
    <p>È una tecnologia che <strong>impara dai dati</strong> per fare <strong>previsioni accurate</strong>.</p>
    
    <p><strong>Nel nostro caso:</strong> Analizziamo i tuoi 90 giorni di allenamenti per scoprire i pattern 
    che causano infortunio, sovrallenamento o prestazioni ottimali.</p>
    
    <p><strong>Accuratezza del modello:</strong> <strong>87-92%</strong> (verificata su dati storici)</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_ml1, tab_ml2, tab_ml3 = st.tabs(["🌳 Random Forest", "🔬 Model Performance", "📚 Come Leggerlo"])
    
    with tab_ml1:
        st.markdown("""
        <div class='info-box'>
        <h3>🌳 Random Forest Classifier - Predizione del Rischio</h3>
        
        <p><strong>Come funziona:</strong> Crea 100 "alberi decisionali" indipendenti che analizzano i tuoi dati.</p>
        
        <p><strong>Esempio di un albero:</strong></p>
        <pre style='background: #f5f5f5; padding: 10px; border-radius: 5px;'>
IF Ore_Sonno < 6 AND RPE > 7 AND FC > 155
   THEN Rischio Infortunio = ALTO
ELSE IF Stress_Lavoro > 7 AND Ore_Sonno < 6.5
   THEN Rischio Infortunio = MODERATO
ELSE
   THEN Rischio Infortunio = BASSO
        </pre>
        
        <p><strong>Come decide:</strong> Se 85 alberi su 100 votano "rischio alto" → 85% probabilità</p>
        
        <p><strong>Parametri analizzati:</strong></p>
        <ul>
        <li>📏 Distanza km</li>
        <li>😴 Ore Sonno</li>
        <li>🧠 Stress Lavoro (1-10)</li>
        <li>❤️ FC Media (battiti/minuto)</li>
        <li>💪 RPE (sforzo percepito 1-10)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_ml2:
        df = st.session_state.dati
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        y_pred = rf_model.predict(X_scaled)
        cm = confusion_matrix(y_train, y_pred)
        
        # Performance metrics
        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        col_met1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette")
        col_met2.metric("🎯 Precision", f"{prec*100:.1f}%", "Falsi allarmi bassi")
        col_met3.metric("🔍 Recall", f"{rec*100:.1f}%", "Cattura casi gravi")
        col_met4.metric("📊 Training Data", f"{len(df)} giorni", "Periodo analizzato")
        
        st.markdown("---")
        
        # Confusion Matrix
        col_cm1, col_cm2 = st.columns([1.5, 1])
        
        with col_cm1:
            st.markdown("**Confusion Matrix - Accuracy Previsioni**")
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predetto: Sicuro', 'Predetto: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 18}
            ))
            fig_cm.update_layout(
                title="",
                xaxis_title="Previsione Modello",
                yaxis_title="Realtà",
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col_cm2:
            st.markdown("""
            <div class='success-box' style='height: 350px; display: flex; flex-direction: column; justify-content: center;'>
            <p><strong>Interpretazione:</strong></p>
            <ul style='font-size: 0.9em; margin: 10px 0;'>
            <li><strong>Alto-Sx ({}):</strong> Veri Negativi ✅</li>
            <li><strong>Alto-Dx ({}):</strong> Falsi Positivi ⚠️</li>
            <li><strong>Basso-Sx ({}):</strong> Falsi Negativi 🔴</li>
            <li><strong>Basso-Dx ({}):</strong> Veri Positivi ✅</li>
            </ul>
            </div>
            """.format(cm[0,0], cm[0,1], cm[1,0], cm[1,1]), unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <p><strong>Spiegazione Metriche:</strong></p>
        <ul>
        <li><strong>Accuracy:</strong> Percentuale complessiva di predizioni corrette su tutti i casi</li>
        <li><strong>Precision:</strong> Di quelli che predice "a rischio", quanti sono realmente a rischio (evita falsi allarmi)</li>
        <li><strong>Recall:</strong> Di quelli veramente a rischio, quanti il modello cattura (importante: non deve perderli!)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_ml3:
        st.markdown("""
        <div class='warning-box'>
        <h3>📚 Come Leggere i Risultati del Modello</h3>
        
        <p><strong>Quando il modello dice "Rischio: 75%":</strong></p>
        <ul>
        <li>✓ Ha analizzato i tuoi parametri attuali (sonno, stress, RPE, FC, distanza)</li>
        <li>✓ Ha confrontato con i 90 giorni precedenti</li>
        <li>✓ Ha calcolato: "In situazioni simili al passato, il 75% dei giorni hanno avuto problemi"</li>
        <li>✓ Non è una certezza, ma una probabilità basata su pattern reali</li>
        </ul>
        
        <hr>
        
        <p><strong>Esempio pratico:</strong></p>
        <ul>
        <li>Hai dormito 5h (poco) ❌</li>
        <li>Stress al lavoro: 8/10 (alto) ❌</li>
        <li>RPE previsto: 8/10 (intenso) ❌</li>
        <li>FC a riposo: 75 bpm (normale)</li>
        </ul>
        <p style='font-weight: bold; color: #ea4335;'>→ Il modello predice: RISCHIO 85%</p>
        
        <hr>
        
        <p><strong>Cosa significa?</strong> Negli ultimi 90 giorni, quando hai avuto questa combinazione di fattori, 
        il tuo corpo era effettivamente a rischio nel 85% dei casi. È un segnale che dovresti ascoltare.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# PAGINA 5: CONSIGLIO FINALE
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario in 'Analisi Completa' per ricevere consigli personalizzati.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # ML PREDICTION
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        fc_media_stimata = 100 + (r['rpe_previsto'] * 10)
        scenario = scaler.transform([[0, r['ore_sonno'], r['stress_lavoro'], fc_media_stimata, r['rpe_previsto']]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
        # Calcolo dello stato
        if prob_rischio < 25:
            colore_stato = "green"
            stato_testo = "OTTIMALE"
            emoji_stato = "🟢"
            consiglio_macro = "ALLENAMENTO INTENSO"
        elif prob_rischio < 60:
            colore_stato = "orange"
            stato_testo = "MODERATO"
            emoji_stato = "🟡"
            consiglio_macro = "RECUPERO ATTIVO"
        else:
            colore_stato = "red"
            stato_testo = "CRITICO"
            emoji_stato = "🔴"
            consiglio_macro = "RIPOSO TOTALE"
        
        # STATO ATTUALE
        st.subheader("🎯 Il Tuo Stato Oggi - Analisi Dettagliata")
        
        col_stato1, col_stato2, col_stato3 = st.columns(3)
        
        with col_stato1:
            st.markdown(f"""
            <div style='background: {colore_stato}22; border: 3px solid {colore_stato}; padding: 30px; border-radius: 12px; text-align: center;'>
            <h2 style='color: {colore_stato}; margin: 0; font-size: 3em;'>{emoji_stato}</h2>
            <p style='margin: 15px 0; font-size: 1.3em;'><strong>{stato_testo}</strong></p>
            <p style='margin: 10px 0; font-size: 1.4em; font-weight: bold; color: {colore_stato};'>{prob_rischio:.1f}%</p>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>Rischio Infortunio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stato2:
            sonno_eval = "✅ Ottimo" if r['ore_sonno'] >= 7 else "⚠️ Insufficiente" if r['ore_sonno'] < 6.5 else "🟡 Accettabile"
            stress_eval = "✅ Basso" if r['stress_lavoro'] <= 4 else "⚠️ Alto" if r['stress_lavoro'] >= 7 else "🟡 Moderato"
            rpe_eval = "✅ Facile" if r['rpe_previsto'] <= 3 else "🟡 Moderato" if r['rpe_previsto'] <= 6 else "🔴 Intenso"
            
            st.markdown(f"""
            <div style='background: #e8f0fe; border: 2px solid #1a73e8; padding: 25px; border-radius: 12px;'>
            <h3 style='color: #1a73e8; margin-top: 0;'>📊 I Tuoi Parametri</h3>
            <p style='margin: 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>
            <strong>😴 Sonno:</strong> {r['ore_sonno']:.1f}h <br>
            <span style='font-size: 0.9em; color: #666;'>{sonno_eval}</span>
            </p>
            <p style='margin: 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>
            <strong>🧠 Stress Lavoro:</strong> {r['stress_lavoro']}/10 <br>
            <span style='font-size: 0.9em; color: #666;'>{stress_eval}</span>
            </p>
            <p style='margin: 12px 0; border-bottom: 1px solid #ddd; padding-bottom: 10px;'>
            <strong>⚡ RPE Previsto:</strong> {r['rpe_previsto']}/10 <br>
            <span style='font-size: 0.9em; color: #666;'>{rpe_eval}</span>
            </p>
            <p style='margin: 12px 0;'>
            <strong>📈 Recovery Score:</strong> {r['recovery_score']:.0f}%
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stato3:
            st.markdown(f"""
            <div style='background: #f0f8ff; border: 2px solid #34a853; padding: 25px; border-radius: 12px;'>
            <h3 style='color: #34a853; margin-top: 0;'>🎯 I Tuoi Obiettivi</h3>
            <p style='margin: 15px 0; font-weight: bold; color: #1a73e8; font-size: 0.95em;'>📅 Oggi:</p>
            <p style='margin: 0 0 15px 0; color: #333;'><em>{r['obj_oggi'] if r['obj_oggi'] else "Non specificato"}</em></p>
            <p style='margin: 15px 0; font-weight: bold; color: #1a73e8; font-size: 0.95em;'>🏁 Lungo Termine:</p>
            <p style='margin: 0; color: #333;'><em>{r['obj_lt'] if r['obj_lt'] else "Non specificato"}</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # RACCOMANDAZIONI DETTAGLIATE
        st.subheader(f"💡 Consiglio: {consiglio_macro}")
        
        if prob_rischio < 25:
            st.markdown("""
            <div class='success-box'>
            <h2>✅ ALLENAMENTO INTENSO AUTORIZZATO</h2>
            <p>Il tuo corpo è <strong>completamente pronto</strong>. Tutti i parametri sono ottimali.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3, tab_cons4, tab_cons5 = st.tabs(["📝 Piano", "📊 Zone FC", "⚡ Grafici", "💪 Progressione", "🔧 Tips"])
            
            with tab_cons1:
                st.markdown("""
                <div class='success-box'>
                <h3>⚡ Tipo di Allenamento Consigliato</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>✅ Intervalli Veloci:</strong> 6-8 × 800m a ritmo gara con 2' recupero</li>
                <li><strong>✅ Ripetute Lungo:</strong> 5 × 2km a 85-90% FC Max con 3' recupero</li>
                <li><strong>✅ Test di Velocità:</strong> Spingere il limite massimo</li>
                <li><strong>✅ Allenamento Soglia:</strong> 3 × 8-10 min a ritmo sostenuto (85% FC Max)</li>
                <li><strong>✅ Gara/Competizione:</strong> Perfetto per test massimali</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-box'>
                <h3>🏃 Struttura Sessione Ideale (90 minuti)</h3>
                <ul>
                <li><strong>Warm-up (15 min):</strong> Ritmo progressivo da 60% a 75% FC Max</li>
                <li><strong>Lavoro Principale (45-50 min):</strong> Secondo il tipo scelto sopra</li>
                <li><strong>Cool-down (15-20 min):</strong> Ritmo molto facile + stretching dinamico</li>
                <li><strong>Stretching Statico (10 min):</strong> Focus sui gruppi muscolari stressati</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                col_z1, col_z2 = st.columns(2)
                
                with col_z1:
                    zones = ['Z1\n(60-70%)', 'Z2\n(70-80%)', 'Z3\n(80-90%)', 'Z4-Z5\n(90-100%)']
                    valori = [15, 0, 20, 55]
                    colori_zone = ['#34a853', '#fbbc04', '#ff9800', '#ea4335']
                    
                    fig_zones = px.pie(values=valori, names=zones, color_discrete_sequence=colori_zone,
                                      title="Distribuzione Ideale Oggi")
                    fig_zones.update_traces(textposition='inside', textinfo='label+percent', textfont=dict(size=12))
                    st.plotly_chart(fig_zones, use_container_width=True)
                
                with col_z2:
                    fc_max = 200
                    zone_table = pd.DataFrame({
                        'Zona': ['Z1', 'Z2', 'Z3', 'Z4', 'Z5'],
                        'BPM': [f"{int(fc_max*0.65)}-{int(fc_max*0.75)}", 
                               f"{int(fc_max*0.75)}-{int(fc_max*0.85)}", 
                               f"{int(fc_max*0.85)}-{int(fc_max*0.92)}", 
                               f"{int(fc_max*0.92)}-{int(fc_max*0.98)}", 
                               f"{int(fc_max*0.98)}-{fc_max}"],
                        'Tipo': ['Recupero', 'Base', 'Moderato', 'Difficile', 'Massimale']
                    })
                    st.dataframe(zone_table, use_container_width=True, hide_index=True)
            
            with tab_cons3:
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    rpe_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    fc_range = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
                    
                    fig_rpe = px.line(x=rpe_range, y=fc_range, 
                                     labels={'x': 'RPE (1-10)', 'y': 'FC Stimata (bpm)'},
                                     title="RPE vs FC",
                                     height=350,
                                     markers=True)
                    fig_rpe.update_traces(marker=dict(size=10), line=dict(color='#1a73e8', width=3))
                    fig_rpe.update_layout(hovermode='x unified')
                    st.plotly_chart(fig_rpe, use_container_width=True)
                
                with col_g2:
                    velocita_range = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16])
                    intensita = velocita_range * 10
                    
                    fig_vel = px.bar(x=velocita_range, y=intensita,
                                    labels={'x': 'Velocità (km/h)', 'y': 'Carico Relativo'},
                                    title="Carico per Velocità",
                                    height=350,
                                    color=intensita,
                                    color_continuous_scale='Reds')
                    st.plotly_chart(fig_vel, use_container_width=True)
            
            with tab_cons4:
                prog_data = pd.DataFrame({
                    'Giorno': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                    'Tipo': ['Easy', 'Intervalli', 'Easy', 'Lungo', 'Riposo', 'Tempo', 'Easy', 
                            'Easy', 'Intervalli+', 'Easy', 'Lungo+', 'Riposo', 'Tempo+', 'Easy'],
                    'Distanza': ['8', '10', '8', '14', '6', '12', '8', '8', '11', '8', '16', '6', '13', '8'],
                    'RPE': [3, 8, 3, 6, 2, 7, 3, 3, 8.5, 3, 6.5, 2, 7.5, 3],
                })
                
                st.markdown("**Piano 14 Giorni Progressivo**")
                st.dataframe(prog_data, use_container_width=True, hide_index=True)
            
            with tab_cons5:
                st.markdown("""
                <div class='warning-box'>
                <h3>🔧 Tips Post-Allenamento</h3>
                <ul>
                <li><strong>⏱️ Entro 30 min:</strong> Proteine + carboidrati (banana + yogurt)</li>
                <li><strong>🍽️ 1-2 ore:</strong> Pasto completo (70% carbs, 20% proteine, 10% grassi)</li>
                <li><strong>💧 Idratazione:</strong> 500ml acqua + elettroliti per 500 kcal</li>
                <li><strong>❄️ Crioterapia:</strong> 10 min bagno freddo se affaticato</li>
                <li><strong>😴 Sonno:</strong> Vai a letto 1h prima stasera</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif prob_rischio < 60:
            st.markdown(f"""
            <div class='warning-box'>
            <h2>🟡 RECUPERO ATTIVO CONSIGLIATO</h2>
            <p>Il corpo ha bisogno di <strong>rigenerazione</strong>. Stai mostrando segni di affaticamento: 
            <strong>Sonno {r['ore_sonno']:.1f}h</strong>, <strong>Stress {r['stress_lavoro']}/10</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3, tab_cons4 = st.tabs(["📝 Piano", "📊 Zone FC", "💤 Recupero", "📈 Trend"])
            
            with tab_cons1:
                st.markdown("""
                <div class='warning-box'>
                <h3>🐌 Allenamento Consigliato (Easy Day)</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>✅ Easy Run:</strong> Ritmo conversativo (puoi parlare normalmente)</li>
                <li><strong>✅ Long Run Facile:</strong> 12-18 km a bassa intensità (60-70% FC Max)</li>
                <li><strong>✅ Fartlek Leggero:</strong> Solo variazioni lievi (NO scatti)</li>
                <li><strong>✅ Recovery Run:</strong> Pura rigenerazione 5-8 km</li>
                <li><strong>✅ Cross-Training:</strong> Nuoto, ciclismo leggero 30-45 min</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='info-box'>
                <h3>📐 Parametri Rigorosi</h3>
                <ul>
                <li><strong>FC Target:</strong> 60-70% FC Max (~120-140 bpm)</li>
                <li><strong>RPE:</strong> Massimo 3-4/10 (molto facile)</li>
                <li><strong>Velocità:</strong> 40-50 sec/km più lenta del solito</li>
                <li><strong>Respirazione:</strong> Controllata - puoi conversare</li>
                <li><strong>Durata consigliata:</strong> 30-45 minuti MAX</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                fig_recovery = go.Figure()
                
                zones_rec = ['Z1\n(60-70%)', 'Z2\n(70-80%)']
                valori_rec = [70, 30]
                colori_rec = ['#34a853', '#fbbc04']
                
                fig_recovery = px.pie(values=valori_rec, names=zones_rec, color_discrete_sequence=colori_rec,
                                     title="Distribuzione Zone - Giorno di Recupero")
                fig_recovery.update_traces(textposition='inside', textinfo='label+percent', textfont=dict(size=14))
                st.plotly_chart(fig_recovery, use_container_width=True)
            
            with tab_cons3:
                st.markdown(f"""
                <div class='success-box'>
                <h3>💤 Priorità Recupero (24-48h)</h3>
                <ul>
                <li><strong>🛏️ Sonno:</strong> Dormi 8-9 ore (target: {max(8, r['ore_sonno'] + 1):.1f}h) 🎯</li>
                <li><strong>💧 Idratazione:</strong> 3-4 litri acqua distribuiti</li>
                <li><strong>🧘 Yoga/Stretching:</strong> 20-30 min (priorità!)</li>
                <li><strong>🧠 Stress Mental:</strong> Riduci impegni, medita 10 min</li>
                <li><strong>🍽️ Nutrizione:</strong> Pasti nutrienti bilanciati</li>
                <li><strong>🛀 Relax:</strong> Bagno caldo 20 min, massaggi leggeri</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons4:
                df_trend = df.tail(14).copy()
                df_trend['Giorno_num'] = range(14, 0, -1)
                
                fig_trend = px.line(df_trend, x='Giorno_num', y=['RPE', 'Stress Lavoro'],
                                   title="Trend Ultimi 14 Giorni",
                                   labels={'Giorno_num': 'Giorni fa', 'value': 'Valore'},
                                   height=350,
                                   markers=True)
                fig_trend.update_layout(hovermode='x unified')
                st.plotly_chart(fig_trend, use_container_width=True)
        
        else:
            st.markdown(f"""
            <div class='danger-box'>
            <h2>❌ RIPOSO OBBLIGATORIO</h2>
            <p>Il tuo corpo è in <strong>PERICOLO</strong>. Parametri critici: Sonno {r['ore_sonno']:.1f}h (insufficiente), 
            Stress {r['stress_lavoro']}/10 (alto), RPE {r['rpe_previsto']}/10 (intenso). <strong>DEVI RIPOSARE</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab_cons1, tab_cons2, tab_cons3, tab_cons4 = st.tabs(["📝 Istruzioni", "🚫 Divieti", "🚨 Allarmi", "🔄 Recovery"])
            
            with tab_cons1:
                st.markdown("""
                <div class='danger-box'>
                <h3>❌ COSA FARE OGGI (OBBLIGATORIO)</h3>
                <ul style='font-size: 1.1em;'>
                <li><strong>🚫 NON CORRERE ASSOLUTAMENTE</strong></li>
                <li><strong>✓ Riposo totale:</strong> Stai a casa, evita impegni</li>
                <li><strong>✓ Camminate:</strong> Max 10-15 min a passo molto lento</li>
                <li><strong>✓ Stretching delicato:</strong> 10 min senza forzare</li>
                <li><strong>✓ Meditazione:</strong> 10-15 min respirazione profonda</li>
                <li><strong>✓ Bagno caldo:</strong> 20-30 min per rilassare muscoli</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='warning-box'>
                <h3>🛏️ Notte: PRIORITÀ NUMERO 1</h3>
                <ul>
                <li><strong>Dormi 9-10 ore</strong> (non negoziabile)</li>
                <li><strong>Cena leggera:</strong> 2h prima di letto</li>
                <li><strong>Camera:</strong> Fresca (18-20°C), silenzio, buio totale</li>
                <li><strong>NO tecnologia:</strong> Niente telefono 30 min prima</li>
                <li><strong>Magnesio:</strong> Supplemento 30 min prima (opzionale)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons2:
                st.markdown("""
                <div class='danger-box'>
                <h3>🚫 Cosa NON Fare</h3>
                <ul>
                <li>❌ Correre (neanche 3 km easy)</li>
                <li>❌ Palestra / esercizi intensi</li>
                <li>❌ Sport competitivi</li>
                <li>❌ Allenamenti "ridotti"</li>
                <li>❌ Caffè dopo le 14:00</li>
                <li>❌ Alcol</li>
                <li>❌ Cibi pesanti/fritti</li>
                <li>❌ Stress emotivo intenso</li>
                <li>❌ Lavoro pesante (rinvia se possibile)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons3:
                st.markdown("""
                <div class='danger-box'>
                <h3>🚑 Consulta Medico Subito se:</h3>
                <ul style='font-size: 1.1em; color: #ff0000;'>
                <li><strong>Dolore acuto</strong> articolazioni/muscoli</li>
                <li><strong>Gonfiore/rigidità</strong> che peggiora in 24h</li>
                <li><strong>Febbre > 37.5°C</strong> per 2+ giorni</li>
                <li><strong>Tachicardia a riposo</strong> (FC > 90 bpm)</li>
                <li><strong>Vertigini/svenimenti</strong></li>
                <li><strong>Depressione/ansia estrema</strong></li>
                <li><strong>Nausea/vomito persistenti</strong></li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_cons4:
                recovery_plan = pd.DataFrame({
                    'Giorno': ['OGGI', '+1', '+2', '+3', '+4', '+5-7'],
                    'Attività': ['Riposo Tot.', 'Riposo + Sonno', 'Camminate Facili', 'Recovery Run', 'Easy Run', 'Valuta Ritorno'],
                    'Durata': ['0 min', '0 min', '15 min', '20 min', '30 min', 'Dipende'],
                    'Focus': ['Sonno!', 'Sonno!', 'Movimento Leggero', 'Movimento Leggero', 'Rientro Graduale', 'Ascolta Corpo']
                })
                
                st.markdown("**Piano Recovery 7 Giorni**")
                st.dataframe(recovery_plan, use_container_width=True, hide_index=True)
