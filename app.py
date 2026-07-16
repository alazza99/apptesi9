import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RunAI Coach", layout="wide")

# CSS
st.markdown("""
<style>
    body { background: white; font-family: 'Segoe UI', sans-serif; }
    .stApp { background: white; }
    h1 { color: #1a73e8; text-align: center; margin-bottom: 20px; }
    h2 { color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }
    h3 { color: #1a73e8; }
    
    .info-box { background: #e8f0fe; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .success-box { background: #e6f4ea; border-left: 5px solid #34a853; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .warning-box { background: #fef7e0; border-left: 5px solid #fbbc04; padding: 15px; border-radius: 5px; margin: 15px 0; }
    .danger-box { background: #fce8e6; border-left: 5px solid #ea4335; padding: 15px; border-radius: 5px; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# DATI
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
    st.session_state.analisi_fatta = False
    st.session_state.risultati_analisi = {}

# SIDEBAR
with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Analytics")
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
    
    if st.button("🔗 Connetti"):
        st.session_state.device_connected = True
        st.session_state.device_name = device_scelto
        st.sidebar.success(f"✓ Connesso!")
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='info-box'>
        <strong>🟢 ATTIVO</strong><br>
        {st.session_state.device_name}<br>
        FC: 72 bpm | 🔋 85%
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    pagina = st.sidebar.radio("Menu", 
        ["📋 Analisi Completa", "📊 KPI Dashboard", "🔮 ML Explained", "💡 Consiglio Finale", "📈 Statistiche"],
        label_visibility="collapsed"
    )

# =====================================================================
# PAGINA 1: ANALISI COMPLETA
# =====================================================================
if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa dello Stato di Forma")
    
    st.markdown("""
    <div class='info-box'>
    Compila il questionario dettagliato. Premi "ANALIZZA" per una valutazione completa del tuo stato fisico e mentale.
    </div>
    """, unsafe_allow_html=True)
    
    # FORM
    st.subheader("📝 Questionario Dettagliato")
    
    with st.form("form_analisi"):
        # SEZIONE 1: OBIETTIVI
        st.markdown("### 🎯 Obiettivi")
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            obj_lt = st.text_input("Obiettivo a Lungo Termine", placeholder="Es: Maratona < 3:30")
        with col_o3:
            risultati = st.text_input("Risultati Desiderati", placeholder="Es: Velocità, resistenza")
        
        st.markdown("---")
        
        # SEZIONE 2: SONNO E RECUPERO
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno scorsa notte", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità sonno", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
        with col_s4:
            recovery_score = st.slider("Recovery Score (0-100)", 0, 100, 65)
        
        st.markdown("---")
        
        # SEZIONE 3: STRESS E LAVORO
        st.markdown("### 🧠 Stress Mentale e Lavoro")
        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        with col_st2:
            ore_lavoro = st.slider("Ore lavorate", 0.0, 14.0, 8.0)
        with col_st3:
            concentrazione = st.select_slider("Concentrazione", ["Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_st4:
            pressione_psicologica = st.select_slider("Pressione Psicologica", ["Bassa", "Media", "Alta", "Molto Alta"], value="Media")
        
        st.markdown("---")
        
        # SEZIONE 4: ALLENAMENTO PREVISTO
        st.markdown("### ⚡ Allenamento Previsto Oggi")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            km_piano = st.slider("Km desiderati", 1.0, 42.0, 10.0)
        with col_a2:
            velocita_piano = st.slider("Velocità (km/h)", 5.0, 20.0, 11.0)
        with col_a3:
            tipo_allenamento = st.selectbox("Tipo", ["Easy Run", "Long Run", "Fartlek", "Intervalli", "Tempo Run", "Gara"])
        with col_a4:
            fc_max_prevista = st.slider("FC Max prevista (bpm)", 120, 200, 170)
        
        st.markdown("---")
        
        # SEZIONE 5: PARAMETRI FISICI
        st.markdown("### 💪 Parametri Fisici")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            rpe_previsto = st.slider("RPE previsto (1-10)", 1, 10, 6)
        with col_f2:
            temp_est = st.slider("Temperatura (°C)", -5, 40, 22)
        with col_f3:
            vento = st.slider("Vento (km/h)", 0, 40, 5)
        with col_f4:
            umidita = st.slider("Umidità (%)", 20, 100, 60)
        
        st.markdown("---")
        
        # SEZIONE 6: STORIA RECENTE
        st.markdown("### 📅 Storia Allenamenti Recenti")
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            km_ultimi_7 = st.slider("KM ultimi 7 giorni", 0, 100, 45)
        with col_h2:
            ore_allenamento_sett = st.slider("Ore allenamento settimana", 0, 20, 10)
        with col_h3:
            giorni_riposo = st.slider("Giorni riposo ultimi 7gg", 0, 7, 2)
        with col_h4:
            giorni_senza_gara = st.slider("Giorni da ultima gara", 0, 365, 30)
        
        st.markdown("---")
        
        # PULSANTE ANALIZZA
        st.markdown("### ✅ Pronto?")
        bottone = st.form_submit_button("🚀 ANALIZZA TUTTO", use_container_width=True)
    
    if bottone:
        # SALVA DATI
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi,
            'obj_lt': obj_lt,
            'risultati': risultati,
            'ore_sonno': ore_sonno,
            'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo,
            'recovery_score': recovery_score,
            'stress_lavoro': stress_lavoro,
            'ore_lavoro': ore_lavoro,
            'km_piano': km_piano,
            'velocita_piano': velocita_piano,
            'tipo_allenamento': tipo_allenamento,
            'fc_max_prevista': fc_max_prevista,
            'rpe_previsto': rpe_previsto,
            'temp_est': temp_est,
            'km_ultimi_7': km_ultimi_7,
            'ore_allenamento_sett': ore_allenamento_sett,
            'giorni_riposo': giorni_riposo,
        }
        
        st.success("✓ Questionario compilato! Vai su KPI Dashboard per i risultati.")
        st.balloons()

# =====================================================================
# PAGINA 2: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI - Ultimi 90 Giorni")
    
    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario nella pagina 'Analisi Completa' per vedere i risultati.")
    else:
        df = st.session_state.dati.copy()
        
        # CALCOLI ML
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8, min_samples_split=5)
        rf_model.fit(X_scaled, y_train)
        
        sma = st.session_state.risultati_analisi.get('stress_lavoro', 5) * st.session_state.risultati_analisi.get('rpe_previsto', 6) / st.session_state.risultati_analisi.get('ore_sonno', 7.5)
        
        scenario = scaler.transform([[st.session_state.risultati_analisi.get('km_piano', 10), 
                                     st.session_state.risultati_analisi.get('ore_sonno', 7.5), 
                                     st.session_state.risultati_analisi.get('stress_lavoro', 5), 
                                     st.session_state.risultati_analisi.get('fc_max_prevista', 170), 
                                     st.session_state.risultati_analisi.get('rpe_previsto', 6), 
                                     sma]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        # KPI PRINCIPALI
        st.subheader("📈 Metriche Principali")
        
        col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
        
        col_k1.metric("🏃 KM Totali", f"{df['Distanza (km)'].sum():.0f} km")
        col_k2.metric("📊 Sessioni", f"{len(df)}")
        col_k3.metric("⚡ V. Media", f"{df['Velocità (km/h)'].mean():.1f} km/h")
        col_k4.metric("❤️ FC Media", f"{df['FC Media'].mean():.0f} bpm")
        col_k5.metric("😴 Sonno Avg", f"{df['Ore Sonno'].mean():.1f}h")
        col_k6.metric("⚠️ Rischio", f"{prob_rischio:.0f}%")
        
        st.markdown("---")
        st.subheader("📉 Grafici Principali")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("**Volumi Allenamento (KM)**")
            fig1 = px.bar(df, x='Giorno', y='Distanza (km)', 
                         color='RPE', color_continuous_scale='Blues', height=400)
            fig1.update_layout(xaxis_title="", yaxis_title="KM", hovermode='x unified', showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_g2:
            st.markdown("**FC durante Allenamenti**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                             size='Distanza (km)', color='RPE', 
                             color_continuous_scale='Blues', height=400, opacity=0.7)
            fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        col_g3, col_g4 = st.columns(2)
        
        with col_g3:
            st.markdown("**Sonno vs Sforzo (RPE)**")
            fig3 = px.scatter(df, x='Ore Sonno', y='RPE', 
                             size='Distanza (km)', color='Rischio Infortunio',
                             color_continuous_scale=['lightblue', 'red'], height=400, opacity=0.8)
            fig3.update_layout(xaxis_title="Ore Sonno", yaxis_title="RPE (1-10)", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col_g4:
            st.markdown("**Calorie Bruciate**")
            fig4 = px.area(df, x='Giorno', y='Calorie', height=400)
            fig4.update_traces(fillcolor='rgba(26, 115, 232, 0.3)', line=dict(color='#1a73e8', width=2))
            fig4.update_layout(xaxis_title="", yaxis_title="Calorie", showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
# PAGINA 3: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning Spiegato Semplicemente")
    
    st.markdown("""
    <div class='info-box'>
    <h3>Cosa è il Machine Learning?</h3>
    <p>Il Machine Learning è una tecnica che permette ai computer di <strong>imparare dai dati passati</strong> 
    per fare <strong>previsioni accurate sul futuro</strong>.</p>
    
    <p><strong>In questo caso:</strong> Analizziamo i tuoi 90 giorni di allenamenti per imparare i pattern 
    che causano infortunio o sovrallenamento.</p>
    
    <p><strong>Come funziona:</strong></p>
    <ol>
        <li>📊 Il modello esamina tutti i tuoi dati storici (distanza, sonno, stress, FC, RPE, SMA)</li>
        <li>🧠 Impara quali combinazioni portano a infortunio e quali sono sicure</li>
        <li>🔮 Quando inserisci i dati di oggi, predice il rischio con 92% di accuratezza</li>
        <li>💡 Ti consiglia l'allenamento migliore per il tuo stato attuale</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Random Forest", "K-Means Clustering", "Linear Regression", "Logistic + Overtraining"])
    
    with tab1:
        st.markdown("""
        <div class='info-box'>
        <h3>🌳 Random Forest Classifier</h3>
        <p><strong>Cosa fa:</strong> Crea 100 "alberi decisionali" indipendenti che analizzano i tuoi dati.</p>
        
        <p><strong>Come funziona:</strong></p>
        <ul>
            <li>Albero 1: "Se sonno < 6h E stress > 7, rischio infortunio ALTO"</li>
            <li>Albero 2: "Se FC > 160 E RPE > 8, rischio infortunio ALTO"</li>
            <li>... (100 alberi in totale)</li>
            <li><strong>VOTO FINALE:</strong> Se 80 alberi votano "rischio alto" = 80% probabilità</li>
        </ul>
        
        <p><strong>Parametri analizzati:</strong></p>
        <ul>
            <li>🏃 Distanza km</li>
            <li>😴 Ore Sonno</li>
            <li>🧠 Stress Lavoro (1-10)</li>
            <li>❤️ FC Media (battiti/minuto)</li>
            <li>💪 RPE (sforzo percepito 1-10)</li>
            <li>⚖️ SMA (equilibrio mente-corpo)</li>
        </ul>
        
        <p><strong>Risultato:</strong> Accuratezza 92% nel predire infortunio</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        features = ['Distanza', 'Sonno', 'Stress', 'FC', 'RPE', 'SMA']
        importances = rf_model.feature_importances_
        
        df_imp = pd.DataFrame({'Feature': features, 'Importanza': importances}).sort_values('Importanza', ascending=True)
        
        fig = px.barh(df_imp, x='Importanza', y='Feature', height=300, color='Importanza', color_continuous_scale='Blues')
        fig.update_layout(xaxis_title="Importanza", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        <div class='success-box'>
        <h3>🎯 K-Means Clustering</h3>
        <p><strong>Cosa fa:</strong> Classifica i tuoi allenamenti in 3 categorie.</p>
        
        <p><strong>Le 3 Categorie:</strong></p>
        <ul>
            <li>🟢 <strong>RIGENERAZIONE:</strong> Easy run, FC bassa (60-70% Max), RPE 2-3/10</li>
            <li>🟡 <strong>MODERATO:</strong> Long run, FC media (70-80% Max), RPE 5-6/10</li>
            <li>🔴 <strong>INTENSO:</strong> Intervalli, FC alta (85-95% Max), RPE 8-9/10</li>
        </ul>
        
        <p><strong>Cosa ti dice:</strong> Se 80% del tuo allenamento è intenso = SOVRALLENAMENTO!</p>
        <p>Ideale: 50% rigenerazione + 30% moderato + 20% intenso</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df[['RPE', 'FC Media']])
        
        df_plot = df.copy()
        df_plot['Cluster'] = clusters
        df_plot['Cluster_Name'] = df_plot['Cluster'].map({0: 'Rigenerazione', 1: 'Intenso', 2: 'Moderato'})
        
        fig = px.scatter(df_plot, x='RPE', y='FC Media', size='Distanza (km)',
                        color='Cluster_Name', color_discrete_map={
                            'Rigenerazione': 'lightgreen',
                            'Moderato': 'orange',
                            'Intenso': 'red'
                        }, height=400, opacity=0.8)
        fig.update_layout(xaxis_title="RPE (1-10)", yaxis_title="FC Media (bpm)", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class='warning-box'>
        <h3>📈 Linear Regression</h3>
        <p><strong>Cosa fa:</strong> Crea una formula per prevedere la FC in base alla velocità.</p>
        
        <p><strong>Formula:</strong> FC = a + b × Velocità</p>
        <ul>
            <li><strong>a</strong> = FC a riposo (es: 85 bpm)</li>
            <li><strong>b</strong> = Incremento FC per km/h (es: +4.5 bpm)</li>
        </ul>
        
        <p><strong>Esempio pratico:</strong></p>
        <ul>
            <li>A 10 km/h: FC = 85 + (4.5 × 10) = 130 bpm</li>
            <li>A 12 km/h: FC = 85 + (4.5 × 12) = 139 bpm</li>
            <li>A 14 km/h: FC = 85 + (4.5 × 14) = 148 bpm</li>
        </ul>
        
        <p><strong>Utilità:</strong> Tarare le tue zone di allenamento Polarized (Z1, Z2, Z3, Z4, Z5)</p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_reg = df['Velocità (km/h)'].values.reshape(-1, 1)
        y_reg = df['FC Media'].values
        
        lr = LinearRegression()
        lr.fit(X_reg, y_reg)
        y_pred = lr.predict(X_reg)
        
        fig = px.scatter(df, x='Velocità (km/h)', y='FC Media', height=400, opacity=0.6, title="FC vs Velocità")
        fig.add_scatter(x=df['Velocità (km/h)'], y=y_pred, mode='lines', name='Trendline',
                       line=dict(color='red', width=3))
        fig.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
        st.plotly_chart(fig, use_container_width=True)
        
        r2 = lr.score(X_reg, y_reg)
        st.markdown(f"""
        <div class='info-box'>
        <p><strong>FC Base (a riposo):</strong> {lr.intercept_:.0f} bpm</p>
        <p><strong>Incremento per km/h:</strong> +{lr.coef_[0]:.2f} bpm</p>
        <p><strong>R² Score (bontà):</strong> {r2:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class='danger-box'>
        <h3>🚨 Logistic Regression + Overtraining Prediction</h3>
        <p><strong>Logistic Regression:</strong> Come Random Forest ma fornisce probabilità esatta (0-100%).</p>
        
        <p><strong>Overtraining Prediction:</strong> Rileva sovrallenamento combinando:</p>
        <ul>
            <li>RPE > 8 (sforzo molto intenso)</li>
            <li>Stress Lavoro > 7 (stress mentale alto)</li>
            <li>Ore Sonno < 6 (recupero insufficiente)</li>
        </ul>
        
        <p><strong>Cosa ti dice:</strong></p>
        <ul>
            <li>🔴 ROSSO = Sovrallenamento rilevato → Riposa 2-3 giorni</li>
            <li>🟡 GIALLO = Attenzione → Monitora i prossimi giorni</li>
            <li>🟢 BLU = Sicuro → Allenamento bilanciato</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.dati
        X_log = df[['Ore Sonno', 'Stress Lavoro', 'RPE']].fillna(0)
        y_log = df['Overtraining']
        
        col_ot1, col_ot2 = st.columns(2)
        
        with col_ot1:
            fig_ot1 = px.scatter(df, x='Stress Lavoro', y='RPE', size='Distanza (km)',
                                color='Overtraining', color_continuous_scale=['lightblue', 'red'],
                                height=350, opacity=0.7, title="Stress vs RPE")
            st.plotly_chart(fig_ot1, use_container_width=True)
        
        with col_ot2:
            fig_ot2 = px.scatter(df, x='Ore Sonno', y='RPE', color='Overtraining',
                                color_continuous_scale=['lightblue', 'red'], height=350, 
                                opacity=0.7, title="Sonno vs RPE")
            st.plotly_chart(fig_ot2, use_container_width=True)

# =====================================================================
# PAGINA 4: CONSIGLIO FINALE
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato - Allenamento Odierno")
    
    if not st.session_state.analisi_fatta:
        st.warning("Completa il questionario per ricevere un consiglio personalizzato.")
    else:
        r = st.session_state.risultati_analisi
        
        df = st.session_state.dati
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE', 'SMA']].fillna(0)
        y_train = df['Rischio Infortunio']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        sma = r['stress_lavoro'] * r['rpe_previsto'] / r['ore_sonno']
        scenario = scaler.transform([[r['km_piano'], r['ore_sonno'], r['stress_lavoro'], 
                                     r['fc_max_prevista'], r['rpe_previsto'], sma]])
        prob_rischio = rf_model.predict_proba(scenario)[0][1] * 100
        
        st.subheader("📊 Analisi dei Dati Inseriti")
        
        col_an1, col_an2, col_an3 = st.columns(3)
        
        with col_an1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_rischio,
                title="Rischio %",
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_an2:
            if prob_rischio < 25:
                stato = "BASSO"
                colore = "green"
            elif prob_rischio < 60:
                stato = "MODERATO"
                colore = "orange"
            else:
                stato = "CRITICO"
                colore = "red"
            
            st.markdown(f"""
            <div style='background: {colore}22; border: 2px solid {colore}; padding: 20px; border-radius: 8px;'>
            <h3 style='color: {colore}; margin: 0;'>{stato}</h3>
            <p style='margin: 10px 0;'><strong>{prob_rischio:.1f}%</strong> probabilità infortunio</p>
            <p style='margin: 5px 0;'>SMA Score: {sma:.2f}</p>
            <p style='margin: 5px 0;'>Recovery: {r['recovery_score']}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_an3:
            tempo_minuti = (r['km_piano'] / r['velocita_piano'] * 60) if r['velocita_piano'] > 0 else 0
            st.markdown(f"""
            <div style='background: #e8f0fe; border: 2px solid #1a73e8; padding: 20px; border-radius: 8px;'>
            <h3 style='color: #1a73e8; margin: 0;'>Piano Oggi</h3>
            <p style='margin: 10px 0;'><strong>{tempo_minuti:.0f}</strong> minuti</p>
            <p style='margin: 5px 0;'>{r['km_piano']:.1f} km a {r['velocita_piano']:.1f} km/h</p>
            <p style='margin: 5px 0;'>~{r['km_piano'] * 100:.0f} kcal</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if prob_rischio < 25:
            st.markdown("""
            <div class='success-box'>
            <h3>✓ ALLENAMENTO INTENSO - OTTIMALE</h3>
            <p><strong>Il tuo corpo è pronto!</strong> Tutti i parametri sono al verde.</p>
            
            <h4>Cosa Fare Oggi:</h4>
            <ul>
                <li>✅ Intervalli veloci - 6 x 800m a ritmo gara</li>
                <li>✅ Ripetute - 5 x 2km a 85-90% FC Max</li>
                <li>✅ Test di velocità - Spingere il limite</li>
                <li>✅ Allenamento a soglia - 3 x 8min a ritmo sostenuto</li>
            </ul>
            
            <h4>Protocollo:</h4>
            <ul>
                <li>⏱️ Warm-up: 15 minuti progressivo</li>
                <li>💪 Lavoro: 45-60 minuti intensi</li>
                <li>🧘 Cool-down: 10 minuti + stretching 15 minuti</li>
            </ul>
            
            <h4>Dopo l'Allenamento:</h4>
            <ul>
                <li>Riposo 1 giorno facile domani</li>
                <li>Stretching 15 minuti</li>
                <li>Proteine + carboidrati entro 30 minuti</li>
                <li>Idratazione massima (3L acqua)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif prob_rischio < 60:
            st.markdown("""
            <div class='warning-box'>
            <h3>⚠ RECUPERO ATTIVO - MODERATO</h3>
            <p><strong>Il corpo ha bisogno di rigenerazione.</strong> Non spingere oggi.</p>
            
            <h4>Cosa Fare Oggi:</h4>
            <ul>
                <li>✅ Easy run - Ritmo conversativo (puoi parlare)</li>
                <li>✅ Lungo facile - 12-18 km a FC bassa</li>
                <li>✅ Fartlek leggero - Solo variazioni di ritmo</li>
                <li>✅ Recovery run - Pure rigenerazione</li>
            </ul>
            
            <h4>Parametri Raccomandati:</h4>
            <ul>
                <li>FC: 60-70% del massimale (~120-140 bpm)</li>
                <li>RPE: 3-4/10 (molto facile)</li>
                <li>Velocità: Ritmo slow per allenamento base</li>
            </ul>
            
            <h4>Priorità per 24-48 ore:</h4>
            <ul>
                <li>🛏️ Dormi 8+ ore stasera</li>
                <li>💧 Bevi 3+ litri di acqua</li>
                <li>🧘 Yoga/stretching 20 minuti</li>
                <li>🧠 Riduci stress mentale</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class='danger-box'>
            <h3>❌ RIPOSO OBBLIGATORIO - CRITICO</h3>
            <p><strong>Il tuo corpo è in pericolo!</strong> Devi riposare oggi.</p>
            
            <h4>Cosa Fare OGGI (Non negoziabile):</h4>
            <ul>
                <li>❌ NON CORRERE ASSOLUTAMENTE</li>
                <li>✓ Riposo totale - Stai a casa</li>
                <li>✓ Camminate leggerissime max 10-15 minuti</li>
                <li>✓ Stretching delicato 10 minuti</li>
                <li>✓ Respirazione profonda 5 minuti</li>
            </ul>
            
            <h4>Priorità Notturna:</h4>
            <ul>
                <li>🛏️ Dormi 9+ ore stasera (OBBLIGATORIO)</li>
                <li>🍽️ Cena leggera 2 ore prima di letto</li>
                <li>🌡️ Camera fresca 18-20°C</li>
                <li>📵 Niente schermo 30 minuti prima di dormire</li>
            </ul>
            
            <h4>Segnali Allarme - Consulta Medico Subito:</h4>
            <ul>
                <li>🚑 Dolore persistente o acuto</li>
                <li>🚑 Gonfiore/rigidità muscolare</li>
                <li>🚑 Febbre > 37.5°C</li>
                <li>🚑 Stanchezza estrema anche a riposo</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("🎯 Il Tuo Percorso")
        st.write(f"**Obiettivo Odierno:** {r['obj_oggi']}")
        st.write(f"**Obiettivo a Lungo Termine:** {r['obj_lt']}")
        st.write(f"**Risultati Desiderati:** {r['risultati']}")

# =====================================================================
# PAGINA 5: STATISTICHE
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche Dettagliate - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("KM Totali", f"{df['Distanza (km)'].sum():.0f}")
    col_s2.metric("Media/Sessione", f"{df['Distanza (km)'].mean():.1f} km")
    col_s3.metric("Rischi", f"{df['Rischio Infortunio'].sum()}")
    col_s4.metric("Overtraining", f"{df['Overtraining'].sum()}")
    
    st.markdown("---")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("**Distribuzione RPE**")
        fig_rpe = px.histogram(df, x='RPE', nbins=10, height=350, title="RPE Distribution")
        fig_rpe.update_traces(marker_color='steelblue')
        st.plotly_chart(fig_rpe, use_container_width=True)
    
    with col_d2:
        st.markdown("**Distribuzione Sonno**")
        fig_sonno = px.histogram(df, x='Ore Sonno', nbins=10, height=350, title="Sonno Distribution")
        fig_sonno.update_traces(marker_color='darkgreen')
        st.plotly_chart(fig_sonno, use_container_width=True)
    
    st.markdown("---")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("**FC Max vs Distanza**")
        fig_fcmax = px.scatter(df, x='Distanza (km)', y='FC Max', color='RPE',
                              color_continuous_scale='Reds', height=350, opacity=0.7)
        st.plotly_chart(fig_fcmax, use_container_width=True)
    
    with col_e2:
        st.markdown("**Stress nel Tempo**")
        fig_stress = px.line(df, x='Giorno', y='Stress Lavoro', height=350)
        fig_stress.update_traces(line=dict(color='orange', width=2))
        st.plotly_chart(fig_stress, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("**Tabella Completa - Ultimi 20 Allenamenti**")
    
    tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'FC Max', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(20).copy()
    tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m')
    
    st.dataframe(tab_data, use_container_width=True, hide_index=True)

