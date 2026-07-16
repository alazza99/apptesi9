import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
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
        ["📋 Analisi Completa", "📈 Statistiche", "📊 KPI Dashboard", "💡 Consiglio Finale", "🔮 ML Explained"],
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
# PAGINA 2: STATISTICHE (PRIMA del consiglio)
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
        • Z1-Z2: {}-130 bpm (recupero)<br>
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
        
        # WIDGET PRINCIPALI
        st.subheader("🎯 Valutazione Odierna")
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        with col_kpi1:
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_rischio,
                title="Rischio Infortunio",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 60], 'color': "gray"},
                        {'range': [60, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                },
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col_kpi2:
            fig_recovery = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r['recovery_score'],
                title="Recovery Score",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ]
                },
                number={'suffix': "%"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_recovery.update_layout(height=300)
            st.plotly_chart(fig_recovery, use_container_width=True)
        
        with col_kpi3:
            fig_sma = go.Figure(go.Indicator(
                mode="number+delta",
                value=sma,
                title="SMA Score",
                number={'valueformat': '.1f'},
                delta={'reference': 12, 'relative': False}
            ))
            fig_sma.update_layout(height=300)
            st.plotly_chart(fig_sma, use_container_width=True)
        
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
            <p><strong>Interpret*

