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
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Running Analytics")
    st.markdown("---")
    
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
        <div class='success-box'>
        <strong>🟢 DISPOSITIVO CONNESSO</strong><br>
        <small style='font-size: 0.9em;'>{}</small><br><br>
        <strong>Dati Attuali:</strong><br>
        ❤️ FC: {} bpm<br>
        🔋 Batteria: {}%<br>
        👟 Passi: {:,}<br>
        🔥 Calorie: {} kcal<br><br>
        <small>Sync: {}</small>
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
    <strong>ℹ️ Compila i tuoi parametri odierni.</strong>
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
                         title="Volume Settimanale", height=350,
                         color='Distanza (km)', color_continuous_scale='Blues')
            fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("Incremento graduale ideale: 10% settimanale. Evita picchi improvvisi.")
        
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', 
                            title="Progresso Cumulativo", height=350,
                            markers=True, line_shape='linear')
            fig_cum.update_layout(xaxis_title="Data", yaxis_title="KM Cumulativi")
            st.plotly_chart(fig_cum, use_container_width=True)
            
            st.info(f"Total: {df['Cumulativa'].iloc[-1]:.0f} km in 90 giorni = {df['Cumulativa'].iloc[-1]/90:.1f} km/giorno")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                            size='Distanza (km)', color='RPE',
                            color_continuous_scale='Viridis', height=350, opacity=0.7,
                            title="Efficienza Cardiaca")
            fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info("Punti bassi = cuore più efficiente. Colore = RPE.")
        
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, 
                             title="Sforzo Percepito (RPE)", height=350,
                             color_discrete_sequence=['#1a73e8'])
            fig3.update_layout(xaxis_title="RPE (1-10)", yaxis_title="Giorni")
            
            easy_pct = (df['RPE'] <= 3).sum() / len(df) * 100
            hard_pct = (df['RPE'] >= 7).sum() / len(df) * 100
            
            fig3.add_vline(x=3.5, line_dash="dash", line_color="green")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info(f"{easy_pct:.0f}% Easy + {hard_pct:.0f}% Intenso = Polarized Training")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', 
                              title="Trend Sonno", height=350, markers=True)
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="green")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_sleep, use_container_width=True)
            
            media_sonno = df['Ore Sonno'].mean()
            st.info(f"Media: {media_sonno:.1f}h/notte | Target: 7.5h")
        
        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', 
                            size='Distanza (km)', color='Rischio Infortunio',
                            color_continuous_scale=['lightblue', 'red'],
                            height=350, opacity=0.8,
                            title="Recupero vs Intensità")
            fig4.add_hline(y=7, line_dash="dash", line_color="orange")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="orange")
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
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div style='background: {status_color}15; border: 4px solid {status_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>Rischio Infortunio</strong></p>
            <hr style='border: 1px solid {status_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: Sonno (40p) + Stress (35p) + RPE (30p) + Combo (20p)<br>
            <strong>0-25%:</strong> Sicuro 🟢<br>
            <strong>25-60%:</strong> Moderato 🟡<br>
            <strong>60-100%:</strong> Riposo 🔴
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            rec_emoji = "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "❌"
            rec_color = "#34a853" if recovery_score >= 75 else "#fbbc04" if recovery_score >= 40 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {rec_color}15; border: 4px solid {rec_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {rec_color};'>{rec_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {rec_color};'>{recovery_score:.0f}%</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>Recovery Score</strong></p>
            <hr style='border: 1px solid {rec_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: 100 - |Sonno - 7.5| × 13.33<br>
            <strong>Target sonno:</strong> 7.5h per maximizzare<br>
            <strong>Capacità di recupero</strong>
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            sma_emoji = "✅" if sma < 10 else "⚠️" if sma < 15 else "❌"
            sma_color = "#34a853" if sma < 10 else "#fbbc04" if sma < 15 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {sma_color}15; border: 4px solid {sma_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {sma_color};'>{sma_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {sma_color};'>{sma:.1f}</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>SMA Score</strong></p>
            <hr style='border: 1px solid {sma_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: (Stress × RPE) / Sonno<br>
            <strong>Equilibrio</strong> stress/sforzo vs recupero<br>
            <strong><10:</strong> Bilanciato ✅
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Visualizzazioni")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title="Risk Score",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, 25], 'color': "#e6f4ea"},
                        {'range': [25, 60], 'color': "#fef7e0"},
                        {'range': [60, 100], 'color': "#fce8e6"}
                    ]
                },
                number={'suffix': '%', 'font': {'size': 24}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno\n(h)', 'Stress\n(1-10)', 'RPE\n(1-10)', 'Recovery\n(%)'],
                fill='toself',
                name='Parametri',
                marker=dict(color=status_color)
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                height=360,
                title="Radar Parametri"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"vs 7.5h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Base")

elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Spiegazione Completa")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cos'è il Machine Learning?</h3>
    <p><strong>ML</strong> = Algoritmo che "impara" dai dati storici per fare previsioni su nuovi dati.</p>
    <p><strong>Esempio:</strong> Dai 90 giorni passati impara: "Quando ho sonno<6h + stress>7 + RPE>7 → rischio infortunio". 
    Poi predice: "Oggi hai questi parametri → X% probabilità rischio"</p>
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
            <div class='info-box'>
            <h3>🌳 Random Forest - Come Funziona</h3>
            
            <p><strong>Dataset Storico:</strong> {num_giorni} giorni | {num_rischi} giorni a rischio ({num_rischi/num_giorni*100:.1f}%)</p>
            
            <p><strong>Il Modello:</strong></p>
            <ul>
            <li>Crea 100 alberi decisionali indipendenti</li>
            <li>Ogni albero impara pattern diversi dai dati</li>
            <li>Quando predice, TUTTI gli alberi "votano"</li>
            <li>Il risultato è la media dei voti (probabilità)</li>
            </ul>
            
            <p><strong>Esempio di Albero:</strong></p>
            <pre style='background: #f5f5f5; padding: 15px; border-radius: 8px; font-size: 0.85em;'>
ALBERO 1:
├─ IF Ore_Sonno < 6.5
│  ├─ IF RPE > 7 → RISCHIO ALTO ✓
│  └─ ELSE → MODERATO
└─ ELSE → BASSO

ALBERO 2:
├─ IF Stress > 7
│  ├─ IF Sonno < 6 → RISCHIO ✓
│  └─ ELSE → MODERATO
└─ ELSE → BASSO

... (98 altri alberi)

RISULTATO: Se 70 alberi su 100 votano RISCHIO → 70% probabilità
            </pre>
            
            <p><strong>Parametri Analizzati:</strong></p>
            <ul>
            <li>📏 <strong>Distanza (km)</strong> - volume allenamento</li>
            <li>😴 <strong>Ore Sonno</strong> - recupero notturno</li>
            <li>🧠 <strong>Stress Lavoro (1-10)</strong> - carico mentale</li>
            <li>❤️ <strong>FC Media (bpm)</strong> - intensità cardiaca</li>
            <li>💪 <strong>RPE (1-10)</strong> - sforzo percepito</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Quali Parametri Influenzano Più il Rischio?**")
            
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in imp_data]
            importances_sorted = [x[1] for x in imp_data]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=features_sorted,
                x=[x*100 for x in importances_sorted],
                orientation='h',
                marker=dict(color=importances_sorted, colorscale='Reds', showscale=True),
                text=[f'{x*100:.1f}%' for x in importances_sorted],
                textposition='auto'
            ))
            fig_imp.update_layout(
                title="Feature Importance - Quanto Ogni Parametro Influenza",
                xaxis_title="Importanza (%)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown(f"""
            <div class='success-box'>
            <p><strong>🔴 Parametro Più Importante: {features_sorted[0]} ({importances_sorted[0]*100:.1f}%)</strong></p>
            <p>Questo fattore pesa più di tutti gli altri nel predire il rischio.</p>
            <p><strong>Top 3:</strong></p>
            <ul>
            <li>1. {features_sorted[0]}: {importances_sorted[0]*100:.1f}%</li>
            <li>2. {features_sorted[1]}: {importances_sorted[1]*100:.1f}%</li>
            <li>3. {features_sorted[2]}: {importances_sorted[2]*100:.1f}%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("**Confusion Matrix - Come il Modello Predice Correttamente?**")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predetto: Sicuro', 'Predetto: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 24},
                colorscale='RdYlGn_r'
            ))
            fig_cm.update_layout(
                title="Matrice di Confusione - Accuratezza Previsioni",
                xaxis_title="Previsione Modello",
                yaxis_title="Realtà Storica",
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            <div class='info-box'>
            <h4>Come Leggere la Matrice:</h4>
            <ul>
            <li><strong>{cm[0,0]} (Alto-Sinistra):</strong> Giorni realmente SICURI, predetti CORRETTAMENTE ✅</li>
            <li><strong>{cm[0,1]} (Alto-Destra):</strong> Giorni SICURI predetti come RISCHIO (Falso Allarme) ⚠️</li>
            <li><strong>{cm[1,0]} (Basso-Sinistra):</strong> Giorni A RISCHIO predetti come SICURI (Pericoloso!) 🔴</li>
            <li><strong>{cm[1,1]} (Basso-Destra):</strong> Giorni A RISCHIO, predetti CORRETTAMENTE ✅</li>
            </ul>
            <p><strong>Somma diagonale (alto-sx + basso-dx):</strong> Previsioni corrette totali</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("**Performance del Modello su Dati Storici**")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette globalmente")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Dei predetti a rischio, quanti lo sono davvero")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Dei giorni a rischio, quanti cattura")
            
            st.markdown("---")
            
            st.markdown("""
            <div class='warning-box'>
            <h4>📖 Cosa Significano le Metriche?</h4>
            <ul>
            <li><strong>Accuracy:</strong> Percentuale totale di predizioni corrette (sia sicuri che rischio)</li>
            <li><strong>Precision:</strong> Quando il modello dice "rischio", quanto spesso ha ragione? 
            (Bassa = molti falsi allarmi)</li>
            <li><strong>Recall:</strong> Dei giorni REALMENTE a rischio, quanti il modello cattura? 
            (Bassa = rischia di perdere veri pericoli)</li>
            </ul>
            <p><strong>Balance:</strong> Preferisce falsi allarmi (bassa precision) piuttosto che perdere rischi veri (alta recall)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("**Come il Modello Calcola il Rischio OGGI**")
            
            if st.session_state.analisi_fatta:
                r = st.session_state.risultati_analisi
                
                input_data = np.array([[5, r['ore_sonno'], r['stress_lavoro'], 
                                       100 + r['rpe_previsto']*10, r['rpe_previsto']]])
                input_scaled = scaler.transform(input_data)
                
                prob_rischio = rf_model.predict_proba(input_scaled)[0][1] * 100
                
                st.markdown(f"""
                <div class='info-box'>
                <h3>🧮 Calcolo in Tempo Reale</h3>
                
                <p><strong>I Tuoi Parametri Odierni:</strong></p>
                <ul>
                <li>😴 Sonno: {r['ore_sonno']:.1f}h</li>
                <li>🧠 Stress: {r['stress_lavoro']}/10</li>
                <li>⚡ RPE: {r['rpe_previsto']}/10</li>
                <li>❤️ FC Stimata: {100 + r['rpe_previsto']*10:.0f} bpm</li>
                </ul>
                
                <p><strong>Come Calcola:</strong></p>
                <ol>
                <li>Prende i tuoi 5 parametri</li>
                <li>Li "scala" per normalizzarli (come i dati di training)</li>
                <li>Li passa a TUTTI i 100 alberi</li>
                <li>Conta i "voti" per RISCHIO</li>
                <li>Divide il numero di voti per 100 = probabilità %</li>
                </ol>
                
                <p style='font-size: 1.3em; font-weight: bold; color: #1a73e8;'>
                📊 Risultato: {prob_rischio:.1f}% probabilità di rischio infortunio
                </p>
                </div>
                """, unsafe_allow_html=True)
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_prob = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_rischio,
                        title="ML Prediction",
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ea4335" if prob_rischio >= 60 else "#fbbc04" if prob_rischio >= 25 else "#34a853"},
                            'steps': [
                                {'range': [0, 25], 'color': "#e6f4ea"},
                                {'range': [25, 60], 'color': "#fef7e0"},
                                {'range': [60, 100], 'color': "#fce8e6"}
                            ]
                        },
                        number={'suffix': '%', 'font': {'size': 24}}
                    ))
                    fig_prob.update_layout(height=350)
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with col_g2:
                    votes_rischio = int(prob_rischio)
                    votes_safe = 100 - votes_rischio
                    
                    fig_votes = go.Figure(data=[
                        go.Bar(name='Rischio', x=['Voti Alberi'], y=[votes_rischio], marker_color='#ea4335'),
                        go.Bar(name='Sicuro', x=['Voti Alberi'], y=[votes_safe], marker_color='#34a853')
                    ])
                    fig_votes.update_layout(
                        barmode='stack',
                        title=f"Voti dei 100 Alberi (simulato)",
                        yaxis_title="Numero Alberi",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_votes, use_container_width=True)
            
            else:
                st.warning("⚠️ Completa il questionario per vedere il calcolo personalizzato.")
    
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
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
            status_emoji = "🟢"
            title = "ALLENAMENTO INTENSO AUTORIZZATO"
            color = "#34a853"
        elif risk_score < 60:
            status_emoji = "🟡"
            title = "RECUPERO ATTIVO CONSIGLIATO"
            color = "#fbbc04"
        else:
            status_emoji = "🔴"
            title = "RIPOSO OBBLIGATORIO"
            color = "#ea4335"
        
        st.markdown(f"""
        <div style='background: {color}15; border: 4px solid {color}; border-radius: 20px; padding: 35px; text-align: center;'>
        <h2 style='color: {color}; margin: 0;'>{status_emoji} {title}</h2>
        <p style='font-size: 1.3em; color: {color}; margin: 10px 0; font-weight: bold;'>Rischio Infortunio: {risk_score:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Analisi Parametri vs Media (90 giorni)")
        
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
        st.subheader("📈 Grafici Trend - Ultimi 30 Giorni")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        
        df_recent = df.tail(30).copy()
        
        with col_g1:
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True,
                              title="Sonno Trend", line_shape='spline')
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", 
                              annotation_text=f"Oggi: {r['ore_sonno']:.1f}h")
            fig_sonno.add_hline(y=7.5, line_dash="dot", line_color="green", 
                              annotation_text="Target: 7.5h")
            st.plotly_chart(fig_sonno, use_container_width=True)
        
        with col_g2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True,
                            title="RPE Trend", line_shape='spline')
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red",
                            annotation_text=f"Oggi: {r['rpe_previsto']}/10")
            st.plotly_chart(fig_rpe, use_container_width=True)
        
        with col_g3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True,
                               title="Stress Trend", line_shape='spline')
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red",
                               annotation_text=f"Oggi: {r['stress_lavoro']}/10")
            st.plotly_chart(fig_stress, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Grafici Analitici")
        
        col_g4, col_g5 = st.columns(2)
        
        with col_g4:
            fig_scatter = px.scatter(df_recent, x='Ore Sonno', y='RPE', 
                                    size='Distanza (km)', color='FC Media',
                                    color_continuous_scale='Viridis',
                                    height=350, opacity=0.7,
                                    title="Relazione Sonno-RPE-FC")
            fig_scatter.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            fig_scatter.add_vline(x=r['ore_sonno'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_g5:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df['Ore Sonno'], name='Sonno 90gg', marker_color='#1a73e8'))
            fig_box.add_trace(go.Box(y=[r['ore_sonno']], name='Oggi', marker_color='#ea4335'))
            fig_box.update_layout(height=350, title="Sonno: Oggi vs Storico")
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Raccomandazioni")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if risk_score < 25:
                st.markdown("""
                <div class='success-box'>
                <h3>✅ Corpo Pronto - Allenamento Intenso</h3>
                <h4>⚡ Cosa Puoi Fare:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Intervalli Veloci:</strong> 6-8 × 800m (2' recupero)</li>
                <li><strong>Tempo Run:</strong> 3 × 10min a ritmo sostenuto</li>
                <li><strong>Ripetute:</strong> 5 × 2km (3' recupero)</li>
                <li><strong>Test Velocità:</strong> Perfetto per misurare progressi</li>
                <li><strong>Gara:</strong> Condizioni ideali</li>
                </ul>
                <h4>📋 Struttura Ideale (90 min):</h4>
                <ul style='font-size: 0.95em;'>
                <li>Warm-up 15': Progressivo 60%→75% FC Max</li>
                <li>Lavoro 45': Secondo tipo scelto</li>
                <li>Cool-down 15': Ritmo facile</li>
                <li>Stretching 10': Focus muscoli stressati</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown(f"""
                <div class='warning-box'>
                <h3>🟡 Recupero Attivo - Corpo Affaticato</h3>
                <h4>🐌 Allenamento Consigliato:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Easy Run:</strong> Ritmo conversativo 30-40 min</li>
                <li><strong>Recovery Run:</strong> 5-8 km per mobilità</li>
                <li><strong>Long Run Facile:</strong> 12-15 km a 60-70% FC Max</li>
                <li><strong>Cross-Training:</strong> Nuoto/ciclismo leggero 30-45 min</li>
                </ul>
                <h4>💤 PRIORITÀ SONNO:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Target:</strong> {max(8, r['ore_sonno'] + 1):.1f}h stasera</li>
                <li><strong>Idratazione:</strong> 3-4 litri acqua</li>
                <li><strong>Stretching:</strong> 20-30 minuti</li>
                <li><strong>Riposo psichico:</strong> Riduci stress lavoro</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='danger-box'>
                <h3>🔴 Riposo Totale - OBBLIGATORIO</h3>
                <h4>❌ Non Correre Oggi:</h4>
                <ul style='font-size: 1em;'>
                <li>No allenamenti intensi o moderati</li>
                <li>No palestra o esercizi</li>
                <li>No sport competitivi</li>
                <li>✓ Max: camminate 15 min</li>
                <li>✓ Max: stretching delicato</li>
                </ul>
                <h4>🛏️ PRIORITÀ ASSOLUTA:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Sonno:</strong> 9-10 ore STASERA</li>
                <li><strong>Recupero:</strong> Riposo totale</li>
                <li><strong>Nutrizione:</strong> Pasti leggeri nutrienti</li>
                <li><strong>Stress:</strong> Riduci impegni</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col_rec2:
            if risk_score < 25:
                st.markdown("""
                <div class='info-box'>
                <h3>🔄 Post-Allenamento</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>Entro 30 min:</strong> Proteine + carboidrati (banana + yogurt)</li>
                <li><strong>1-2 ore:</strong> Pasto completo (70% carbs, 20% proteine, 10% grassi)</li>
                <li><strong>Idratazione:</strong> 500ml acqua per 500 kcal bruciate</li>
                <li><strong>Sonno:</strong> Dormi 1h prima del solito</li>
                </ul>
                <h3>⏰ Prossimi 3 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>DOMANI:</strong> Easy run 6-8 km ritmo conversativo</li>
                <li><strong>+2:</strong> Riposo attivo o cross-training leggero</li>
                <li><strong>+3:</strong> Allenamento moderato (Fartlek/Long easy)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown("""
                <div class='info-box'>
                <h3>💤 Priorità Recupero 24-48h</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>Sonno:</strong> Dormi 1-2h più del solito</li>
                <li><strong>Yoga/Stretching:</strong> 20-30 minuti dedicati</li>
                <li><strong>Bagno caldo:</strong> 20 minuti rilassamento</li>
                <li><strong>Massaggi:</strong> Se disponibile, leggeri</li>
                <li><strong>Meditazione:</strong> 10-15 minuti stress relief</li>
                </ul>
                <h3>📋 Piano 3 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>OGGI:</strong> Easy run facile</li>
                <li><strong>+1:</strong> Riposo con stretching</li>
                <li><strong>+2:</strong> Recovery run 5-8 km</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='info-box'>
                <h3>🚑 Monitoraggio Salute</h3>
                <ul style='font-size: 0.95em;'>
                <li>Dolore acuto → <strong>MEDICO</strong></li>
                <li>Febbre > 37.5°C → <strong>MEDICO</strong></li>
                <li>Tachicardia FC > 90 bpm → <strong>MEDICO</strong></li>
                <li>Vertigini/svenimenti → <strong>MEDICO</strong></li>
                <li>Depressione estrema → <strong>MEDICO</strong></li>
                </ul>
                <h3>📋 Piano Recovery 7 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>OGGI:</strong> Riposo totale - Sonno 9-10h</li>
                <li><strong>+1:</strong> Riposo - Camminate max 15 min</li>
                <li><strong>+2:</strong> Camminate 20 min + stretching</li>
                <li><strong>+3:</strong> Recovery run 20-30 min</li>
                <li><strong>+4:</strong> Easy run 30-40 min</li>
                <li><strong>+5-7:</strong> Valuta ritorno graduale</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Riepilogo KPI Odierno")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Riposo', 'Recovery', 'SMA', 'Risk'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10",
                      f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': [
                "✅" if r['ore_sonno'] >= 7 else "⚠️" if r['ore_sonno'] >= 6.5 else "🔴",
                "✅" if r['stress_lavoro'] <= 5 else "⚠️" if r['stress_lavoro'] <= 7 else "🔴",
                "✅" if r['rpe_previsto'] <= 5 else "⚠️" if r['rpe_previsto'] <= 7 else "🔴",
                "✅" if r['fc_riposo'] <= 65 else "⚠️",
                "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "🔴",
                "✅" if sma < 10 else "⚠️" if sma < 15 else "🔴",
                "✅" if risk_score < 25 else "⚠️" if risk_score < 60 else "🔴"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)import streamlit as st
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
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Running Analytics")
    st.markdown("---")
    
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
        <div class='success-box'>
        <strong>🟢 DISPOSITIVO CONNESSO</strong><br>
        <small style='font-size: 0.9em;'>{}</small><br><br>
        <strong>Dati Attuali:</strong><br>
        ❤️ FC: {} bpm<br>
        🔋 Batteria: {}%<br>
        👟 Passi: {:,}<br>
        🔥 Calorie: {} kcal<br><br>
        <small>Sync: {}</small>
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
    <strong>ℹ️ Compila i tuoi parametri odierni.</strong>
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
                         title="Volume Settimanale", height=350,
                         color='Distanza (km)', color_continuous_scale='Blues')
            fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.info("Incremento graduale ideale: 10% settimanale. Evita picchi improvvisi.")
        
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', 
                            title="Progresso Cumulativo", height=350,
                            markers=True, line_shape='linear')
            fig_cum.update_layout(xaxis_title="Data", yaxis_title="KM Cumulativi")
            st.plotly_chart(fig_cum, use_container_width=True)
            
            st.info(f"Total: {df['Cumulativa'].iloc[-1]:.0f} km in 90 giorni = {df['Cumulativa'].iloc[-1]/90:.1f} km/giorno")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', 
                            size='Distanza (km)', color='RPE',
                            color_continuous_scale='Viridis', height=350, opacity=0.7,
                            title="Efficienza Cardiaca")
            fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.info("Punti bassi = cuore più efficiente. Colore = RPE.")
        
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, 
                             title="Sforzo Percepito (RPE)", height=350,
                             color_discrete_sequence=['#1a73e8'])
            fig3.update_layout(xaxis_title="RPE (1-10)", yaxis_title="Giorni")
            
            easy_pct = (df['RPE'] <= 3).sum() / len(df) * 100
            hard_pct = (df['RPE'] >= 7).sum() / len(df) * 100
            
            fig3.add_vline(x=3.5, line_dash="dash", line_color="green")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info(f"{easy_pct:.0f}% Easy + {hard_pct:.0f}% Intenso = Polarized Training")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ore di Sonno**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', 
                              title="Trend Sonno", height=350, markers=True)
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="green")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_sleep, use_container_width=True)
            
            media_sonno = df['Ore Sonno'].mean()
            st.info(f"Media: {media_sonno:.1f}h/notte | Target: 7.5h")
        
        with col2:
            st.markdown("**Sonno vs Sforzo**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', 
                            size='Distanza (km)', color='Rischio Infortunio',
                            color_continuous_scale=['lightblue', 'red'],
                            height=350, opacity=0.8,
                            title="Recupero vs Intensità")
            fig4.add_hline(y=7, line_dash="dash", line_color="orange")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="orange")
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
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div style='background: {status_color}15; border: 4px solid {status_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>Rischio Infortunio</strong></p>
            <hr style='border: 1px solid {status_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: Sonno (40p) + Stress (35p) + RPE (30p) + Combo (20p)<br>
            <strong>0-25%:</strong> Sicuro 🟢<br>
            <strong>25-60%:</strong> Moderato 🟡<br>
            <strong>60-100%:</strong> Riposo 🔴
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k2:
            rec_emoji = "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "❌"
            rec_color = "#34a853" if recovery_score >= 75 else "#fbbc04" if recovery_score >= 40 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {rec_color}15; border: 4px solid {rec_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {rec_color};'>{rec_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {rec_color};'>{recovery_score:.0f}%</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>Recovery Score</strong></p>
            <hr style='border: 1px solid {rec_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: 100 - |Sonno - 7.5| × 13.33<br>
            <strong>Target sonno:</strong> 7.5h per maximizzare<br>
            <strong>Capacità di recupero</strong>
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_k3:
            sma_emoji = "✅" if sma < 10 else "⚠️" if sma < 15 else "❌"
            sma_color = "#34a853" if sma < 10 else "#fbbc04" if sma < 15 else "#ea4335"
            
            st.markdown(f"""
            <div style='background: {sma_color}15; border: 4px solid {sma_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {sma_color};'>{sma_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {sma_color};'>{sma:.1f}</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>SMA Score</strong></p>
            <hr style='border: 1px solid {sma_color}30; margin: 15px 0;'>
            <p style='font-size: 0.9em; color: #999; margin: 0;'>
            Formula: (Stress × RPE) / Sonno<br>
            <strong>Equilibrio</strong> stress/sforzo vs recupero<br>
            <strong><10:</strong> Bilanciato ✅
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Visualizzazioni")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title="Risk Score",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, 25], 'color': "#e6f4ea"},
                        {'range': [25, 60], 'color': "#fef7e0"},
                        {'range': [60, 100], 'color': "#fce8e6"}
                    ]
                },
                number={'suffix': '%', 'font': {'size': 24}}
            ))
            fig_gauge.update_layout(height=360)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno\n(h)', 'Stress\n(1-10)', 'RPE\n(1-10)', 'Recovery\n(%)'],
                fill='toself',
                name='Parametri',
                marker=dict(color=status_color)
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                height=360,
                title="Radar Parametri"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"vs 7.5h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Base")

elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Spiegazione Completa")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Cos'è il Machine Learning?</h3>
    <p><strong>ML</strong> = Algoritmo che "impara" dai dati storici per fare previsioni su nuovi dati.</p>
    <p><strong>Esempio:</strong> Dai 90 giorni passati impara: "Quando ho sonno<6h + stress>7 + RPE>7 → rischio infortunio". 
    Poi predice: "Oggi hai questi parametri → X% probabilità rischio"</p>
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
            <div class='info-box'>
            <h3>🌳 Random Forest - Come Funziona</h3>
            
            <p><strong>Dataset Storico:</strong> {num_giorni} giorni | {num_rischi} giorni a rischio ({num_rischi/num_giorni*100:.1f}%)</p>
            
            <p><strong>Il Modello:</strong></p>
            <ul>
            <li>Crea 100 alberi decisionali indipendenti</li>
            <li>Ogni albero impara pattern diversi dai dati</li>
            <li>Quando predice, TUTTI gli alberi "votano"</li>
            <li>Il risultato è la media dei voti (probabilità)</li>
            </ul>
            
            <p><strong>Esempio di Albero:</strong></p>
            <pre style='background: #f5f5f5; padding: 15px; border-radius: 8px; font-size: 0.85em;'>
ALBERO 1:
├─ IF Ore_Sonno < 6.5
│  ├─ IF RPE > 7 → RISCHIO ALTO ✓
│  └─ ELSE → MODERATO
└─ ELSE → BASSO

ALBERO 2:
├─ IF Stress > 7
│  ├─ IF Sonno < 6 → RISCHIO ✓
│  └─ ELSE → MODERATO
└─ ELSE → BASSO

... (98 altri alberi)

RISULTATO: Se 70 alberi su 100 votano RISCHIO → 70% probabilità
            </pre>
            
            <p><strong>Parametri Analizzati:</strong></p>
            <ul>
            <li>📏 <strong>Distanza (km)</strong> - volume allenamento</li>
            <li>😴 <strong>Ore Sonno</strong> - recupero notturno</li>
            <li>🧠 <strong>Stress Lavoro (1-10)</strong> - carico mentale</li>
            <li>❤️ <strong>FC Media (bpm)</strong> - intensità cardiaca</li>
            <li>💪 <strong>RPE (1-10)</strong> - sforzo percepito</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Quali Parametri Influenzano Più il Rischio?**")
            
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in imp_data]
            importances_sorted = [x[1] for x in imp_data]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=features_sorted,
                x=[x*100 for x in importances_sorted],
                orientation='h',
                marker=dict(color=importances_sorted, colorscale='Reds', showscale=True),
                text=[f'{x*100:.1f}%' for x in importances_sorted],
                textposition='auto'
            ))
            fig_imp.update_layout(
                title="Feature Importance - Quanto Ogni Parametro Influenza",
                xaxis_title="Importanza (%)",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown(f"""
            <div class='success-box'>
            <p><strong>🔴 Parametro Più Importante: {features_sorted[0]} ({importances_sorted[0]*100:.1f}%)</strong></p>
            <p>Questo fattore pesa più di tutti gli altri nel predire il rischio.</p>
            <p><strong>Top 3:</strong></p>
            <ul>
            <li>1. {features_sorted[0]}: {importances_sorted[0]*100:.1f}%</li>
            <li>2. {features_sorted[1]}: {importances_sorted[1]*100:.1f}%</li>
            <li>3. {features_sorted[2]}: {importances_sorted[2]*100:.1f}%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("**Confusion Matrix - Come il Modello Predice Correttamente?**")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predetto: Sicuro', 'Predetto: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 24},
                colorscale='RdYlGn_r'
            ))
            fig_cm.update_layout(
                title="Matrice di Confusione - Accuratezza Previsioni",
                xaxis_title="Previsione Modello",
                yaxis_title="Realtà Storica",
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            <div class='info-box'>
            <h4>Come Leggere la Matrice:</h4>
            <ul>
            <li><strong>{cm[0,0]} (Alto-Sinistra):</strong> Giorni realmente SICURI, predetti CORRETTAMENTE ✅</li>
            <li><strong>{cm[0,1]} (Alto-Destra):</strong> Giorni SICURI predetti come RISCHIO (Falso Allarme) ⚠️</li>
            <li><strong>{cm[1,0]} (Basso-Sinistra):</strong> Giorni A RISCHIO predetti come SICURI (Pericoloso!) 🔴</li>
            <li><strong>{cm[1,1]} (Basso-Destra):</strong> Giorni A RISCHIO, predetti CORRETTAMENTE ✅</li>
            </ul>
            <p><strong>Somma diagonale (alto-sx + basso-dx):</strong> Previsioni corrette totali</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("**Performance del Modello su Dati Storici**")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette globalmente")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Dei predetti a rischio, quanti lo sono davvero")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Dei giorni a rischio, quanti cattura")
            
            st.markdown("---")
            
            st.markdown("""
            <div class='warning-box'>
            <h4>📖 Cosa Significano le Metriche?</h4>
            <ul>
            <li><strong>Accuracy:</strong> Percentuale totale di predizioni corrette (sia sicuri che rischio)</li>
            <li><strong>Precision:</strong> Quando il modello dice "rischio", quanto spesso ha ragione? 
            (Bassa = molti falsi allarmi)</li>
            <li><strong>Recall:</strong> Dei giorni REALMENTE a rischio, quanti il modello cattura? 
            (Bassa = rischia di perdere veri pericoli)</li>
            </ul>
            <p><strong>Balance:</strong> Preferisce falsi allarmi (bassa precision) piuttosto che perdere rischi veri (alta recall)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("**Come il Modello Calcola il Rischio OGGI**")
            
            if st.session_state.analisi_fatta:
                r = st.session_state.risultati_analisi
                
                input_data = np.array([[5, r['ore_sonno'], r['stress_lavoro'], 
                                       100 + r['rpe_previsto']*10, r['rpe_previsto']]])
                input_scaled = scaler.transform(input_data)
                
                prob_rischio = rf_model.predict_proba(input_scaled)[0][1] * 100
                
                st.markdown(f"""
                <div class='info-box'>
                <h3>🧮 Calcolo in Tempo Reale</h3>
                
                <p><strong>I Tuoi Parametri Odierni:</strong></p>
                <ul>
                <li>😴 Sonno: {r['ore_sonno']:.1f}h</li>
                <li>🧠 Stress: {r['stress_lavoro']}/10</li>
                <li>⚡ RPE: {r['rpe_previsto']}/10</li>
                <li>❤️ FC Stimata: {100 + r['rpe_previsto']*10:.0f} bpm</li>
                </ul>
                
                <p><strong>Come Calcola:</strong></p>
                <ol>
                <li>Prende i tuoi 5 parametri</li>
                <li>Li "scala" per normalizzarli (come i dati di training)</li>
                <li>Li passa a TUTTI i 100 alberi</li>
                <li>Conta i "voti" per RISCHIO</li>
                <li>Divide il numero di voti per 100 = probabilità %</li>
                </ol>
                
                <p style='font-size: 1.3em; font-weight: bold; color: #1a73e8;'>
                📊 Risultato: {prob_rischio:.1f}% probabilità di rischio infortunio
                </p>
                </div>
                """, unsafe_allow_html=True)
                
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    fig_prob = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_rischio,
                        title="ML Prediction",
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ea4335" if prob_rischio >= 60 else "#fbbc04" if prob_rischio >= 25 else "#34a853"},
                            'steps': [
                                {'range': [0, 25], 'color': "#e6f4ea"},
                                {'range': [25, 60], 'color': "#fef7e0"},
                                {'range': [60, 100], 'color': "#fce8e6"}
                            ]
                        },
                        number={'suffix': '%', 'font': {'size': 24}}
                    ))
                    fig_prob.update_layout(height=350)
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with col_g2:
                    votes_rischio = int(prob_rischio)
                    votes_safe = 100 - votes_rischio
                    
                    fig_votes = go.Figure(data=[
                        go.Bar(name='Rischio', x=['Voti Alberi'], y=[votes_rischio], marker_color='#ea4335'),
                        go.Bar(name='Sicuro', x=['Voti Alberi'], y=[votes_safe], marker_color='#34a853')
                    ])
                    fig_votes.update_layout(
                        barmode='stack',
                        title=f"Voti dei 100 Alberi (simulato)",
                        yaxis_title="Numero Alberi",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_votes, use_container_width=True)
            
            else:
                st.warning("⚠️ Completa il questionario per vedere il calcolo personalizzato.")
    
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
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
            status_emoji = "🟢"
            title = "ALLENAMENTO INTENSO AUTORIZZATO"
            color = "#34a853"
        elif risk_score < 60:
            status_emoji = "🟡"
            title = "RECUPERO ATTIVO CONSIGLIATO"
            color = "#fbbc04"
        else:
            status_emoji = "🔴"
            title = "RIPOSO OBBLIGATORIO"
            color = "#ea4335"
        
        st.markdown(f"""
        <div style='background: {color}15; border: 4px solid {color}; border-radius: 20px; padding: 35px; text-align: center;'>
        <h2 style='color: {color}; margin: 0;'>{status_emoji} {title}</h2>
        <p style='font-size: 1.3em; color: {color}; margin: 10px 0; font-weight: bold;'>Rischio Infortunio: {risk_score:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📊 Analisi Parametri vs Media (90 giorni)")
        
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
        st.subheader("📈 Grafici Trend - Ultimi 30 Giorni")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        
        df_recent = df.tail(30).copy()
        
        with col_g1:
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True,
                              title="Sonno Trend", line_shape='spline')
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", 
                              annotation_text=f"Oggi: {r['ore_sonno']:.1f}h")
            fig_sonno.add_hline(y=7.5, line_dash="dot", line_color="green", 
                              annotation_text="Target: 7.5h")
            st.plotly_chart(fig_sonno, use_container_width=True)
        
        with col_g2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True,
                            title="RPE Trend", line_shape='spline')
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red",
                            annotation_text=f"Oggi: {r['rpe_previsto']}/10")
            st.plotly_chart(fig_rpe, use_container_width=True)
        
        with col_g3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True,
                               title="Stress Trend", line_shape='spline')
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red",
                               annotation_text=f"Oggi: {r['stress_lavoro']}/10")
            st.plotly_chart(fig_stress, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Grafici Analitici")
        
        col_g4, col_g5 = st.columns(2)
        
        with col_g4:
            fig_scatter = px.scatter(df_recent, x='Ore Sonno', y='RPE', 
                                    size='Distanza (km)', color='FC Media',
                                    color_continuous_scale='Viridis',
                                    height=350, opacity=0.7,
                                    title="Relazione Sonno-RPE-FC")
            fig_scatter.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            fig_scatter.add_vline(x=r['ore_sonno'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_g5:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=df['Ore Sonno'], name='Sonno 90gg', marker_color='#1a73e8'))
            fig_box.add_trace(go.Box(y=[r['ore_sonno']], name='Oggi', marker_color='#ea4335'))
            fig_box.update_layout(height=350, title="Sonno: Oggi vs Storico")
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Raccomandazioni")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if risk_score < 25:
                st.markdown("""
                <div class='success-box'>
                <h3>✅ Corpo Pronto - Allenamento Intenso</h3>
                <h4>⚡ Cosa Puoi Fare:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Intervalli Veloci:</strong> 6-8 × 800m (2' recupero)</li>
                <li><strong>Tempo Run:</strong> 3 × 10min a ritmo sostenuto</li>
                <li><strong>Ripetute:</strong> 5 × 2km (3' recupero)</li>
                <li><strong>Test Velocità:</strong> Perfetto per misurare progressi</li>
                <li><strong>Gara:</strong> Condizioni ideali</li>
                </ul>
                <h4>📋 Struttura Ideale (90 min):</h4>
                <ul style='font-size: 0.95em;'>
                <li>Warm-up 15': Progressivo 60%→75% FC Max</li>
                <li>Lavoro 45': Secondo tipo scelto</li>
                <li>Cool-down 15': Ritmo facile</li>
                <li>Stretching 10': Focus muscoli stressati</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown(f"""
                <div class='warning-box'>
                <h3>🟡 Recupero Attivo - Corpo Affaticato</h3>
                <h4>🐌 Allenamento Consigliato:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Easy Run:</strong> Ritmo conversativo 30-40 min</li>
                <li><strong>Recovery Run:</strong> 5-8 km per mobilità</li>
                <li><strong>Long Run Facile:</strong> 12-15 km a 60-70% FC Max</li>
                <li><strong>Cross-Training:</strong> Nuoto/ciclismo leggero 30-45 min</li>
                </ul>
                <h4>💤 PRIORITÀ SONNO:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Target:</strong> {max(8, r['ore_sonno'] + 1):.1f}h stasera</li>
                <li><strong>Idratazione:</strong> 3-4 litri acqua</li>
                <li><strong>Stretching:</strong> 20-30 minuti</li>
                <li><strong>Riposo psichico:</strong> Riduci stress lavoro</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='danger-box'>
                <h3>🔴 Riposo Totale - OBBLIGATORIO</h3>
                <h4>❌ Non Correre Oggi:</h4>
                <ul style='font-size: 1em;'>
                <li>No allenamenti intensi o moderati</li>
                <li>No palestra o esercizi</li>
                <li>No sport competitivi</li>
                <li>✓ Max: camminate 15 min</li>
                <li>✓ Max: stretching delicato</li>
                </ul>
                <h4>🛏️ PRIORITÀ ASSOLUTA:</h4>
                <ul style='font-size: 1em;'>
                <li><strong>Sonno:</strong> 9-10 ore STASERA</li>
                <li><strong>Recupero:</strong> Riposo totale</li>
                <li><strong>Nutrizione:</strong> Pasti leggeri nutrienti</li>
                <li><strong>Stress:</strong> Riduci impegni</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col_rec2:
            if risk_score < 25:
                st.markdown("""
                <div class='info-box'>
                <h3>🔄 Post-Allenamento</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>Entro 30 min:</strong> Proteine + carboidrati (banana + yogurt)</li>
                <li><strong>1-2 ore:</strong> Pasto completo (70% carbs, 20% proteine, 10% grassi)</li>
                <li><strong>Idratazione:</strong> 500ml acqua per 500 kcal bruciate</li>
                <li><strong>Sonno:</strong> Dormi 1h prima del solito</li>
                </ul>
                <h3>⏰ Prossimi 3 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>DOMANI:</strong> Easy run 6-8 km ritmo conversativo</li>
                <li><strong>+2:</strong> Riposo attivo o cross-training leggero</li>
                <li><strong>+3:</strong> Allenamento moderato (Fartlek/Long easy)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown("""
                <div class='info-box'>
                <h3>💤 Priorità Recupero 24-48h</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>Sonno:</strong> Dormi 1-2h più del solito</li>
                <li><strong>Yoga/Stretching:</strong> 20-30 minuti dedicati</li>
                <li><strong>Bagno caldo:</strong> 20 minuti rilassamento</li>
                <li><strong>Massaggi:</strong> Se disponibile, leggeri</li>
                <li><strong>Meditazione:</strong> 10-15 minuti stress relief</li>
                </ul>
                <h3>📋 Piano 3 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>OGGI:</strong> Easy run facile</li>
                <li><strong>+1:</strong> Riposo con stretching</li>
                <li><strong>+2:</strong> Recovery run 5-8 km</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='info-box'>
                <h3>🚑 Monitoraggio Salute</h3>
                <ul style='font-size: 0.95em;'>
                <li>Dolore acuto → <strong>MEDICO</strong></li>
                <li>Febbre > 37.5°C → <strong>MEDICO</strong></li>
                <li>Tachicardia FC > 90 bpm → <strong>MEDICO</strong></li>
                <li>Vertigini/svenimenti → <strong>MEDICO</strong></li>
                <li>Depressione estrema → <strong>MEDICO</strong></li>
                </ul>
                <h3>📋 Piano Recovery 7 Giorni</h3>
                <ul style='font-size: 0.95em;'>
                <li><strong>OGGI:</strong> Riposo totale - Sonno 9-10h</li>
                <li><strong>+1:</strong> Riposo - Camminate max 15 min</li>
                <li><strong>+2:</strong> Camminate 20 min + stretching</li>
                <li><strong>+3:</strong> Recovery run 20-30 min</li>
                <li><strong>+4:</strong> Easy run 30-40 min</li>
                <li><strong>+5-7:</strong> Valuta ritorno graduale</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Riepilogo KPI Odierno")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Riposo', 'Recovery', 'SMA', 'Risk'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10",
                      f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': [
                "✅" if r['ore_sonno'] >= 7 else "⚠️" if r['ore_sonno'] >= 6.5 else "🔴",
                "✅" if r['stress_lavoro'] <= 5 else "⚠️" if r['stress_lavoro'] <= 7 else "🔴",
                "✅" if r['rpe_previsto'] <= 5 else "⚠️" if r['rpe_previsto'] <= 7 else "🔴",
                "✅" if r['fc_riposo'] <= 65 else "⚠️",
                "✅" if recovery_score >= 75 else "⚠️" if recovery_score >= 40 else "🔴",
                "✅" if sma < 10 else "⚠️" if sma < 15 else "🔴",
                "✅" if risk_score < 25 else "⚠️" if risk_score < 60 else "🔴"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
