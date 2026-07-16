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
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.device_connected = False
    st.session_state.device_name = None
    st.session_state.device_data = {'fc': 72, 'battery': 85}
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
        }
    
    if st.session_state.device_connected:
        st.sidebar.markdown(f"""
        <div class='success-box'>
        <strong>🟢 CONNESSO</strong><br>
        <small>{st.session_state.device_name}</small>
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
    <strong>ℹ️ Come funziona:</strong> Compila i parametri odierni. Il sistema analizzerà i tuoi ultimi 90 giorni per darti consigli personalizzati.
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
        st.success("✓ Analisi completata!")

# =====================================================================
# PAGINA 2: STATISTICHE
# =====================================================================
elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche - Ultimi 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    st.subheader("📊 KPI Principali")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    km_totali = df['Distanza (km)'].sum()
    sessioni = len(df)
    media_km = df['Distanza (km)'].mean()
    giorni_rischio = df['Rischio Infortunio'].sum()
    
    col_m1.metric("🏃 KM Totali", f"{km_totali:.0f} km", "3 mesi")
    col_m2.metric("📊 Sessioni", f"{sessioni}")
    col_m3.metric("📐 Media/Sessione", f"{media_km:.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{giorni_rischio}")
    
    st.markdown("---")
    
    st.subheader("📉 Grafici Analitici")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
        df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
        
        fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', title="KM per Settimana", height=350)
        fig1.update_traces(marker_color='#1a73e8')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_g2:
        fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)', 
                         color='RPE', color_continuous_scale='Blues', height=350, opacity=0.7,
                         title="Efficienza Cardiaca")
        st.plotly_chart(fig2, use_container_width=True)
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        fig3 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)', 
                         color='Rischio Infortunio', color_continuous_scale=['lightblue', 'red'], 
                         height=350, opacity=0.8, title="Sonno vs Sforzo")
        fig3.add_hline(y=7, line_dash="dash", line_color="orange")
        fig3.add_vline(x=6.5, line_dash="dash", line_color="orange")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_g4:
        fig4 = px.histogram(df, x='RPE', nbins=9, title="Distribuzione RPE", height=350)
        fig4.update_traces(marker_color='steelblue')
        st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # CALCOLI SEMPLICI
        risk_score = min(100, 
            (40 if r['ore_sonno'] < 6 else 25 if r['ore_sonno'] < 6.5 else 10) +
            (35 if r['stress_lavoro'] >= 8 else 20 if r['stress_lavoro'] >= 6 else 5) +
            (30 if r['rpe_previsto'] >= 8 else 15 if r['rpe_previsto'] >= 6 else 5) +
            (20 if r['ore_sonno'] < 6.5 and r['stress_lavoro'] >= 7 and r['rpe_previsto'] >= 7 else 0)
        )
        
        recovery_score = max(0, 100 - abs(r['ore_sonno'] - 7.5) * 13.33)
        sma = (r['stress_lavoro'] * r['rpe_previsto']) / r['ore_sonno'] if r['ore_sonno'] > 0 else 0
        
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
        
        st.markdown(f"<h3 style='text-align: center; color: {status_color};'>{status_emoji} {status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        # KPI CARDS BELLE
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div style='background: {status_color}15; border: 4px solid {status_color}; border-radius: 20px; padding: 40px; text-align: center;'>
            <p style='font-size: 3em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 2.8em; margin: 15px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.2em; color: #666; margin: 10px 0;'><strong>Rischio Infortunio</strong></p>
            <hr style='border: 1px solid #ddd; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>Analizzato da ML</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #f5f5f5; border-left: 5px solid #1a73e8; padding: 15px; border-radius: 8px; margin-top: 15px;'>
            <p style='margin: 0; font-size: 0.85em; color: #666;'><strong>COSA SIGNIFICA</strong></p>
            <p style='margin: 8px 0; font-size: 0.8em; color: #999;'>
            • 0-25%: Completamente sicuro<br>
            • 25-60%: Attenzione al recupero<br>
            • 60-100%: Riposo obbligatorio
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            rec_emoji = "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "❌"
            rec_color = "#34a853" if recovery_score >= 75 else "#fbbc04" if recovery_score >= 40 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {rec_color}15; border: 4px solid {rec_color}; border-radius: 20px; padding: 40px; text-align: center;'>
            <p style='font-size: 3em; margin: 0; color: {rec_color};'>{rec_emoji}</p>
            <p style='font-size: 2.8em; margin: 15px 0; font-weight: bold; color: {rec_color};'>{recovery_score:.0f}%</p>
            <p style='font-size: 1.2em; color: #666; margin: 10px 0;'><strong>Recovery Score</strong></p>
            <hr style='border: 1px solid #ddd; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>Capacità di recupero</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #f5f5f5; border-left: 5px solid #34a853; padding: 15px; border-radius: 8px; margin-top: 15px;'>
            <p style='margin: 0; font-size: 0.85em; color: #666;'><strong>COME SI CALCOLA</strong></p>
            <p style='margin: 8px 0; font-size: 0.8em; color: #999;'>
            Formula: 100 - |Ore_Sonno - 7.5| × 13.33<br>
            Target: 7.5 ore per massimizzare recupero
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            sma_emoji = "✅" if sma < 10 else "⚠️" if sma < 15 else "❌"
            sma_color = "#34a853" if sma < 10 else "#fbbc04" if sma < 15 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {sma_color}15; border: 4px solid {sma_color}; border-radius: 20px; padding: 40px; text-align: center;'>
            <p style='font-size: 3em; margin: 0; color: {sma_color};'>{sma_emoji}</p>
            <p style='font-size: 2.8em; margin: 15px 0; font-weight: bold; color: {sma_color};'>{sma:.1f}</p>
            <p style='font-size: 1.2em; color: #666; margin: 10px 0;'><strong>SMA Score</strong></p>
            <hr style='border: 1px solid #ddd; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>Stress-Multiplicativo-Acuto</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background: #f5f5f5; border-left: 5px solid #fbbc04; padding: 15px; border-radius: 8px; margin-top: 15px;'>
            <p style='margin: 0; font-size: 0.85em; color: #666;'><strong>FORMULA</strong></p>
            <p style='margin: 8px 0; font-size: 0.8em; color: #999;'>
            (Stress × RPE) / Ore_Sonno<br>
            Balancia stress/sforzo con recupero
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"Δ {r['ore_sonno']-7.5:+.1f}h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10", "Previsto")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Oggi")
        
        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            df_recent = df.tail(30).copy()
            fig_t1 = px.line(df_recent, y='RPE', title="RPE Trend", height=300, markers=True)
            fig_t1.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_t1, use_container_width=True)
        
        with col_t2:
            fig_t2 = px.line(df_recent, y='Ore Sonno', title="Sonno Trend", height=300, markers=True)
            fig_t2.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_t2, use_container_width=True)

# =====================================================================
# PAGINA 4: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Random Forest")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Come Funziona il Modello</h3>
    <p>Analizza i tuoi 90 giorni di dati per scoprire i pattern che causano infortunio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.dati.copy()
    
    # ADDESTRAMENTO
    try:
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
        
        tab1, tab2, tab3, tab4 = st.tabs(["🌳 Modello", "📊 Feature Importance", "🔬 Confusione Matrix", "📈 Performance"])
        
        with tab1:
            st.markdown("""
            <div class='info-box'>
            <h3>🌳 Random Forest - 100 Alberi Decisionali</h3>
            
            <p><strong>Idea:</strong> Crea 100 alberi indipendenti che votano il risultato.</p>
            
            <p><strong>Esempio Semplificato:</strong></p>
            <pre style='background: #f5f5f5; padding: 15px; border-radius: 5px;'>
            ALBERO 1: Se Sonno < 6 E RPE > 7 → RISCHIO
            ALBERO 2: Se Stress > 7 E Sonno < 6.5 → RISCHIO
            ALBERO 3: Se FC > 160 E Sonno < 6 → RISCHIO
            ...
            RISULTATO FINALE: Se >50 alberi votano RISCHIO → 50%+ probabilità
            </pre>
            
            <p><strong>Parametri Analizzati:</strong></p>
            <ul>
            <li>📏 Distanza km - volume allenamento</li>
            <li>😴 Ore Sonno - recupero</li>
            <li>🧠 Stress Lavoro - carico psichico</li>
            <li>❤️ FC Media - intensità cardiaca</li>
            <li>💪 RPE - sforzo percepito</li>
            </ul>
            
            <p><strong>Dati di Training:</strong> {len(df)} giorni storici</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Quali Fattori Influenzano Più il Rischio?**")
            
            # Dataframe per il grafico
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in imp_data]
            importances_sorted = [x[1] for x in imp_data]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=features_sorted,
                x=importances_sorted,
                orientation='h',
                marker=dict(color='#1a73e8')
            ))
            fig_imp.update_layout(
                title="Importanza dei Parametri",
                xaxis_title="Importanza (%)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown(f"""
            <div class='success-box'>
            <p><strong>Fattore più importante: {features_sorted[0]} ({importances_sorted[0]*100:.1f}%)</strong></p>
            <p>Questo parametro influenza più di tutti gli altri le probabilità di infortunio nel modello.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("**Confusion Matrix - Come il Modello Predice**")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Pred: Sicuro', 'Pred: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues'
            ))
            fig_cm.update_layout(
                title="",
                xaxis_title="Previsione Modello",
                yaxis_title="Realtà Storica",
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            <div class='info-box'>
            <p><strong>{cm[0,0]}</strong> (alto-sx): Giorni SICURI predetti CORRETTAMENTE ✅</p>
            <p><strong>{cm[0,1]}</strong> (alto-dx): Giorni SICURI predetti come RISCHIO (Falsi Allarmi)</p>
            <p><strong>{cm[1,0]}</strong> (basso-sx): Giorni A RISCHIO predetti come SICURI (Pericoloso!) 🔴</p>
            <p><strong>{cm[1,1]}</strong> (basso-dx): Giorni A RISCHIO predetti CORRETTAMENTE ✅</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("**Metriche di Performance**")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Corrette complessivamente")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Falsi allarmi bassi")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Cattura rischi veri")
            col_m4.metric("📊 Dataset", f"{len(df)}", "Giorni analizzati")
            
            st.markdown("""
            <div class='warning-box'>
            <p><strong>Accuracy:</strong> % corrette su TUTTI i casi</p>
            <p><strong>Precision:</strong> Dei predetti "a rischio", quanti lo sono davvero (basso = falsi allarmi)</p>
            <p><strong>Recall:</strong> Dei giorni REALMENTE a rischio, quanti il modello cattura (basso = pericoloso!)</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Errore nel modello ML: {str(e)}")

# =====================================================================
# PAGINA 5: CONSIGLIO FINALE
# =====================================================================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima.")
    else:
        r = st.session_state.risultati_analisi
        df = st.session_state.dati.copy()
        
        # CALCOLI
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
            status = "OTTIMALE"
            color = "#34a853"
            emoji = "🟢"
            title = "✅ ALLENAMENTO INTENSO AUTORIZZATO"
        elif risk_score < 60:
            status = "MODERATO"
            color = "#fbbc04"
            emoji = "🟡"
            title = "🟡 RECUPERO ATTIVO CONSIGLIATO"
        else:
            status = "CRITICO"
            color = "#ea4335"
            emoji = "🔴"
            title = "🔴 RIPOSO OBBLIGATORIO"
        
        st.markdown(f"""
        <div style='background: {color}15; border: 4px solid {color}; border-radius: 20px; padding: 35px; text-align: center;'>
        <h2 style='color: {color}; margin: 0;'>{emoji} {status}</h2>
        <p style='font-size: 1.4em; color: {color}; margin: 10px 0; font-weight: bold;'>{title}</p>
        <p style='font-size: 1.1em; color: #666; margin: 0;'>Rischio Infortunio: <strong>{risk_score:.0f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔍 Analisi Parametri vs Media (90 giorni)")
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            sonno_badge = "⬇️ SOTTO" if sonno_vs_media < -0.5 else "⬆️ SOPRA" if sonno_vs_media > 0.5 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border: 2px solid #1a73e8; padding: 20px; border-radius: 12px; text-align: center;'>
            <p style='margin: 0; font-size: 1.2em; font-weight: bold; color: #1a73e8;'>{sonno_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold;'>{r['ore_sonno']:.1f}h</p>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>vs media {media_sonno_90:.1f}h</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a2:
            stress_badge = "⬇️ SOTTO" if stress_vs_media < -1 else "⬆️ SOPRA" if stress_vs_media > 1 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border: 2px solid #1a73e8; padding: 20px; border-radius: 12px; text-align: center;'>
            <p style='margin: 0; font-size: 1.2em; font-weight: bold; color: #1a73e8;'>{stress_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold;'>{r['stress_lavoro']}/10</p>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>vs media {media_stress_90:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a3:
            rpe_badge = "⬇️ SOTTO" if rpe_vs_media < -1 else "⬆️ SOPRA" if rpe_vs_media > 1 else "➡️ PARI"
            st.markdown(f"""
            <div style='background: #f5f5f5; border: 2px solid #1a73e8; padding: 20px; border-radius: 12px; text-align: center;'>
            <p style='margin: 0; font-size: 1.2em; font-weight: bold; color: #1a73e8;'>{rpe_badge}</p>
            <p style='margin: 10px 0; font-size: 2em; font-weight: bold;'>{r['rpe_previsto']}/10</p>
            <p style='margin: 0; font-size: 0.9em; color: #666;'>vs media {media_rpe_90:.1f}/10</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Grafici - Cosa Vedi nel Trend")
        
        tab1, tab2, tab3 = st.tabs(["📈 Sonno", "⚡ RPE", "🧠 Stress"])
        
        with tab1:
            df_recent = df.tail(30).copy()
            fig_sonno = px.line(df_recent, y='Ore Sonno', title="Sonno Ultimi 30 Giorni", height=350, markers=True)
            fig_sonno.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="Target")
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", annotation_text="Oggi")
            st.plotly_chart(fig_sonno, use_container_width=True)
            
            st.markdown(f"""
            **Cosa vedi:** La linea blu mostra il sonno ultimi 30 giorni.
            - **Verde (7.5h):** Target ideale per recupero
            - **Rossa ({r['ore_sonno']:.1f}h):** Sonno di oggi
            
            **Analisi:** Hai dormito {abs(sonno_vs_media):.1f}h {'**meno**' if sonno_vs_media < 0 else '**più**'} della media.
            {'❌ Questo AUMENTA il rischio!' if sonno_vs_media < -0.5 else '✅ Buon segno!' if sonno_vs_media > 0.5 else ''}
            """)
        
        with tab2:
            df_recent = df.tail(30).copy()
            fig_rpe = px.line(df_recent, y='RPE', title="RPE Ultimi 30 Giorni", height=350, markers=True)
            fig_rpe.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderato")
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="blue", annotation_text="Oggi")
            st.plotly_chart(fig_rpe, use_container_width=True)
            
            st.markdown(f"""
            **Cosa vedi:** La linea blu mostra l'intensità percepita ultimi 30 giorni.
            - **Arancione (5/10):** Soglia facile→moderato
            - **Blu ({r['rpe_previsto']}/10):** RPE previsto oggi
            
            **Analisi:** RPE {'**superiore**' if rpe_vs_media > 1 else '**inferiore**' if rpe_vs_media < -1 else '**in linea**'} alla media.
            """)
        
        with tab3:
            df_recent = df.tail(30).copy()
            fig_stress = px.line(df_recent, y='Stress Lavoro', title="Stress Ultimi 30 Giorni", height=350, markers=True)
            fig_stress.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Medio")
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red", annotation_text="Oggi")
            st.plotly_chart(fig_stress, use_container_width=True)
            
            st.markdown(f"""
            **Cosa vedi:** La linea gialla mostra lo stress lavorativo ultimi 30 giorni.
            - **Arancione (5/10):** Livello medio
            - **Rossa ({r['stress_lavoro']}/10):** Stress di oggi
            
            **Analisi:** Stress {'**più alto**' if stress_vs_media > 1 else '**più basso**' if stress_vs_media < -1 else '**normale**'} rispetto alla media.
            """)
        
        st.markdown("---")
        st.subheader("💡 Raccomandazione Dettagliata")
        
        if risk_score < 25:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ ALLENAMENTO INTENSO - AUTORIZZATO</h3>
            
            <p>Tutti i tuoi parametri sono OTTIMALI. Il corpo è completamente pronto per sforzi massimali.</p>
            
            <h4>⚡ Cosa Puoi Fare:</h4>
            <ul>
            <li><strong>Intervalli veloci:</strong> 6-8 × 800m a ritmo gara con 2' di recupero</li>
            <li><strong>Tempo run:</strong> 3 × 10 minuti a ritmo sostenuto (85% FC Max)</li>
            <li><strong>Ripetute lungo:</strong> 5 × 2km con 3' di recupero</li>
            <li><strong>Test di velocità:</strong> Perfetto per valutare i progressi</li>
            <li><strong>Gara:</strong> Condizioni ideali per competere</li>
            </ul>
            
            <h4>📋 Struttura Allenamento (90 min):</h4>
            <ul>
            <li>Warm-up 15 min: Ritmo progressivo 60%→75% FC Max</li>
            <li>Lavoro principale 45-50 min: Secondo tipo scelto</li>
            <li>Cool-down 15-20 min: Ritmo facile + stretching</li>
            <li>Stretching statico 10 min: Focus muscoli stressati</li>
            </ul>
            
            <h4>🔄 Recovery Post-Allenamento:</h4>
            <ul>
            <li>Entro 30 min: Proteine + carboidrati (es: banana + yogurt)</li>
            <li>1-2h: Pasto completo (70% carbs, 20% proteine, 10% grassi)</li>
            <li>Idratazione: 500ml acqua + elettroliti per 500 kcal</li>
            <li>Sonno: Dormi 1h prima del solito stasera</li>
            </ul>
            
            <h4>⏰ Prossimi Giorni:</h4>
            <ul>
            <li><strong>Domani:</strong> Easy run 6-8 km a ritmo conversativo (Z1-Z2)</li>
            <li><strong>Dopodomani:</strong> Riposo attivo o cross-training leggero</li>
            <li><strong>Giorno 3:</strong> Allenamento moderato (Fartlek o Long Run facile)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        elif risk_score < 60:
            st.markdown(f"""
            <div class='warning-box'>
            <h3>🟡 RECUPERO ATTIVO - CONSIGLIATO</h3>
            
            <p>Il corpo mostra segni di affaticamento. Hai <strong>sonno {r['ore_sonno']:.1f}h</strong> (vs media {media_sonno_90:.1f}h) 
            e <strong>stress {r['stress_lavoro']}/10</strong> (vs media {media_stress_90:.1f}/10).</p>
            
            <h4>🐌 Cosa Fare Oggi:</h4>
            <ul>
            <li><strong>Easy Run:</strong> Ritmo conversativo 30-45 km (puoi parlare normalmente)</li>
            <li><strong>Long Run Facile:</strong> 12-15 km a 60-70% FC Max</li>
            <li><strong>Recovery Run:</strong> 5-8 km per mobilità e rigenerazione</li>
            <li><strong>Fartlek Leggero:</strong> Solo variazioni lievi, NO scatti</li>
            <li><strong>Cross-Training:</strong> Nuoto o ciclismo leggero 30-45 min</li>
            </ul>
            
            <h4>⚙️ Parametri Rigorosi:</h4>
            <ul>
            <li>FC Target: 60-70% FC Max (~120-140 bpm)</li>
            <li>RPE: Massimo 3-4/10 (molto facile)</li>
            <li>Velocità: 40-50 sec/km più lenta del solito</li>
            <li>Respirazione: Controllata e facile</li>
            </ul>
            
            <h4>💤 Priorità Recupero (24-48h):</h4>
            <ul>
            <li><strong>🛏️ Sonno:</strong> Dormi 8-9 ore stasera (target: {max(8, r['ore_sonno'] + 1):.1f}h)</li>
            <li><strong>💧 Idratazione:</strong> 3-4 litri acqua distribuiti nella giornata</li>
            <li><strong>🧘 Yoga/Stretching:</strong> 20-30 minuti dedicati</li>
            <li><strong>🧠 Stress Mentale:</strong> Riduci impegni, medita 10 minuti</li>
            <li><strong>🍽️ Nutrizione:</strong> Pasti bilanciati con proteine e carboidrati</li>
            <li><strong>🛀 Relax:</strong> Bagno caldo 20 min, massaggi leggeri</li>
            </ul>
            
            <h4>📋 Attenzione A:</h4>
            <ul>
            <li>⚠️ Sonno insufficiente - prioritaria per ridurre rischio</li>
            <li>⚠️ Stress elevato - riduci carichi di lavoro se possibile</li>
            <li>⚠️ Non ignorare segnali di stanchezza - ascolta il corpo</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class='danger-box'>
            <h3>🔴 RIPOSO OBBLIGATORIO</h3>
            
            <p><strong>⚠️ IL TUO CORPO È IN PERICOLO</strong></p>
            
            <p>Parametri CRITICI:</p>
            <ul>
            <li>😴 Sonno: <strong>{r['ore_sonno']:.1f}h</strong> (insufficiente - media {media_sonno_90:.1f}h)</li>
            <li>🧠 Stress: <strong>{r['stress_lavoro']}/10</strong> (elevato - media {media_stress_90:.1f}/10)</li>
            <li>⚡ RPE: <strong>{r['rpe_previsto']}/10</strong> (intenso - media {media_rpe_90:.1f}/10)</li>
            </ul>
            
            <h4>❌ NON CORRERE OGGI - È OBBLIGATORIO</h4>
            <ul>
            <li>❌ Nessun allenamento intenso o moderato</li>
            <li>❌ Nessuna palestra o esercizi faticosi</li>
            <li>❌ Nessuno sport competitivo</li>
            <li>✓ MAX: Camminate leggerissime 10-15 minuti a passo molto lento</li>
            <li>✓ Stretching delicato 10 minuti SENZA forzare</li>
            </ul>
            
            <h4>🛏️ PRIORITÀ ASSOLUTA - SONNO</h4>
            <ul>
            <li><strong>Dormi 9-10 ore STASERA</strong> (non negoziabile)</li>
            <li>Cena leggera 2 ore prima di andare a letto</li>
            <li>Camera fresca (18-20°C), silenzio totale, buio totale</li>
            <li>NO telefono/tablet 30 minuti prima di dormire</li>
            <li>Magnesio (opzionale): supplemento 30 min prima di letto</li>
            </ul>
            
            <h4>🚫 Cosa NON Fare</h4>
            <ul>
            <li>❌ Correre (neanche 3 km easy)</li>
            <li>❌ Palestra o esercizi intensi</li>
            <li>❌ Sport competitivi</li>
            <li>❌ Caffè dopo le 14:00</li>
            <li>❌ Alcol</li>
            <li>❌ Cibi pesanti o fritti</li>
            <li>❌ Stress emotivo intenso</li>
            <li>❌ Lavoro pesante (rinvia se possibile)</li>
            </ul>
            
            <h4>🚑 CONSULTA MEDICO SUBITO SE:</h4>
            <ul>
            <li>🔴 Dolore acuto in articolazioni/muscoli</li>
            <li>🔴 Gonfiore o rigidità che peggiora in 24h</li>
            <li>🔴 Febbre > 37.5°C per 2+ giorni</li>
            <li>🔴 Tachicardia a riposo (FC > 90 bpm)</li>
            <li>🔴 Vertigini o svenimenti</li>
            <li>🔴 Depressione/ansia estrema</li>
            <li>🔴 Nausea/vomito persistenti</li>
            </ul>
            
            <h4>📋 Piano Recovery 7 Giorni:</h4>
            <ul>
            <li><strong>OGGI:</strong> Riposo totale, sonno 9-10h</li>
            <li><strong>+1:</strong> Riposo + sonno, camminate max 15 min</li>
            <li><strong>+2:</strong> Camminate facili 20 min, stretching</li>
            <li><strong>+3:</strong> Recovery run max 20-30 min</li>
            <li><strong>+4:</strong> Easy run max 30-40 min</li>
            <li><strong>+5-7:</strong> Valuta ritorno graduale (ascolta corpo)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Riepilogo KPI Odierno")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress Lavoro', 'RPE Previsto', 'FC Riposo', 'Recovery Score', 'SMA Score', 'Rischio Infortunio'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10", 
                      f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': [
                "✅ OK" if r['ore_sonno'] >= 7 else "⚠️ Insufficiente",
                "✅ OK" if r['stress_lavoro'] <= 5 else "⚠️ Elevato",
                "✅ OK" if r['rpe_previsto'] <= 5 else "⚠️ Intenso",
                "✅ OK" if r['fc_riposo'] <= 65 else "⚠️ Elevata",
                "✅ OK" if recovery_score >= 75 else "⚠️ Moderato" if recovery_score >= 40 else "❌ Pessimo",
                "✅ OK" if sma < 10 else "⚠️ Moderato" if sma < 15 else "❌ Squilibrato",
                "✅ Sicuro" if risk_score < 25 else "⚠️ Moderato" if risk_score < 60 else "🔴 CRITICO"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
