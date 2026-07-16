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

with st.sidebar:
    st.markdown("# 🏃 RunAI Coach")
    st.markdown("Professional Running Analytics")
    st.markdown("---")
    
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
    <strong>ℹ️ Compila i tuoi parametri odierni. Il sistema analizzerà i dati per creare un consiglio personalizzato.</strong>
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
        st.success("✓ Analisi completata!")

# =====================================================================
# PAGINA 2: STATISTICHE
# =====================================================================
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
    st.subheader("📖 Analisi Dettagliata per Categoria")
    
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
            
            st.info("""
            **Cosa vedi:** Totale km per ogni settimana.
            **Interpretazione:** Incremento graduale ideale (10% settimanale). Evita picchi improvvisi.
            """)
        
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', 
                            title="Progresso Cumulativo", height=350,
                            markers=True, line_shape='linear')
            fig_cum.update_layout(xaxis_title="Data", yaxis_title="KM Cumulativi")
            st.plotly_chart(fig_cum, use_container_width=True)
            
            st.info(f"""
            **Cosa vedi:** Totale km progressivo nel tempo.
            **Total:** {df['Cumulativa'].iloc[-1]:.0f} km in 90 giorni = {df['Cumulativa'].iloc[-1]/90:.1f} km/giorno
            """)
    
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
            
            st.info("""
            **Cosa vedi:** Relazione velocità-FC. Dimensione punto = distanza. Colore = RPE.
            **Buon segno:** Punti bassi = cuore più efficiente a stessa velocità.
            """)
        
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, 
                             title="Sforzo Percepito (RPE)", height=350,
                             color_discrete_sequence=['#1a73e8'])
            fig3.update_layout(xaxis_title="RPE (1-10)", yaxis_title="Giorni")
            
            easy_pct = (df['RPE'] <= 3).sum() / len(df) * 100
            hard_pct = (df['RPE'] >= 7).sum() / len(df) * 100
            
            fig3.add_vline(x=3.5, line_dash="dash", line_color="green", annotation_text="Easy")
            fig3.add_vline(x=6.5, line_dash="dash", line_color="red", annotation_text="Intenso")
            
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info(f"""
            **Cosa vedi:** Distribuzione degli allenamenti per intensità.
            **Analisi:** {easy_pct:.0f}% Easy (✅ buono) + {hard_pct:.0f}% Intenso = Polarized Training
            """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ore di Sonno Giornaliere**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', 
                              title="Trend Sonno", height=350, markers=True)
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="Target 7.5h")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red", annotation_text="Minimo 6.5h")
            st.plotly_chart(fig_sleep, use_container_width=True)
            
            media_sonno = df['Ore Sonno'].mean()
            giorni_insufficiente = (df['Ore Sonno'] < 6.5).sum()
            
            st.info(f"""
            **Media:** {media_sonno:.1f}h/notte
            **Giorni insufficienti (<6.5h):** {giorni_insufficiente} giorni = {giorni_insufficiente/len(df)*100:.0f}%
            """)
        
        with col2:
            st.markdown("**Sonno vs Sforzo (RPE)**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', 
                            size='Distanza (km)', color='Rischio Infortunio',
                            color_continuous_scale=['lightblue', 'red'],
                            height=350, opacity=0.8,
                            title="Recupero vs Intensità")
            fig4.add_hline(y=7, line_dash="dash", line_color="orange", annotation_text="RPE Alto")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="orange", annotation_text="Sonno Basso")
            st.plotly_chart(fig4, use_container_width=True)
            
            giorni_rischio = df['Rischio Infortunio'].sum()
            st.info(f"""
            **Punto rosso = Giorno a rischio infortunio**
            **Giorni a rischio:** {giorni_rischio} su {len(df)} = {giorni_rischio/len(df)*100:.1f}%
            **Formula:** RPE>7 + Sonno<6.5 = PERICOLO
            """)
    
    with tab4:
        st.markdown("**Ultimi 15 Allenamenti**")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno', 'Stress Lavoro']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%y')
        tab_data['Rischio'] = df['Rischio Infortunio'].tail(15).apply(lambda x: '🔴 RISCHIO' if x == 1 else '✅ OK')
        
        st.dataframe(tab_data, use_container_width=True, hide_index=True)

# =====================================================================
# PAGINA 3: KPI DASHBOARD
# =====================================================================
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
        
        st.markdown(f"<h3 style='text-align: center; color: {status_color};'>{status_emoji} Stato: {status_text}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        col_k1, col_k2, col_k3 = st.columns(3)
        
        with col_k1:
            st.markdown(f"""
            <div style='background: {status_color}15; border: 4px solid {status_color}; border-radius: 20px; padding: 45px 20px; text-align: center;'>
            <p style='font-size: 3.5em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 3.2em; margin: 15px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.3em; color: #666; margin: 10px 0;'><strong>Rischio Infortunio</strong></p>
            <hr style='border: 1px solid {status_color}30; margin: 15px 0;'>
            <p style='font-size: 0.95em; color: #999; margin: 0;'>
            0-25%: Sicuro 🟢<br>
            25-60%: Attenzione 🟡<br>
            60-100%: Riposo 🔴
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
            <p style='font-size: 0.95em; color: #999; margin: 0;'>
            Formula: 100 - |Sonno - 7.5| × 13.33<br>
            Target: 7.5h sonno
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
            <p style='font-size: 0.95em; color: #999; margin: 0;'>
            (Stress × RPE) / Sonno<br>
            <10: Bilanciato
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
                title="Radar Parametri Odierni"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📋 Parametri Attuali")
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h", f"vs 7.5h")
        col_p2.metric("🧠 Stress Lavoro", f"{r['stress_lavoro']}/10", "Livello")
        col_p3.metric("⚡ RPE Previsto", f"{r['rpe_previsto']}/10", "Sforzo")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm", "Base")
        
        st.markdown("---")
        st.subheader("📈 Trend Ultimi 30 Giorni")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            df_recent = df.tail(30).copy()
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True,
                              title="Sonno Trend")
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red", 
                              annotation_text=f"Oggi: {r['ore_sonno']:.1f}h")
            st.plotly_chart(fig_sonno, use_container_width=True)
        
        with col_t2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True,
                            title="RPE Trend")
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red",
                            annotation_text=f"Oggi: {r['rpe_previsto']}/10")
            st.plotly_chart(fig_rpe, use_container_width=True)
        
        with col_t3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True,
                               title="Stress Trend")
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red",
                               annotation_text=f"Oggi: {r['stress_lavoro']}/10")
            st.plotly_chart(fig_stress, use_container_width=True)

# =====================================================================
# PAGINA 4: ML EXPLAINED
# =====================================================================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning - Random Forest")
    
    st.markdown("""
    <div class='info-box'>
    <h3>🤖 Modello Predittivo</h3>
    <p>Random Forest analizza 90 giorni di allenamenti per predire il rischio infortunio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        df = st.session_state.dati.copy()
        num_giorni = len(df)
        
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
        
        tab1, tab2, tab3 = st.tabs(["🌳 Come Funziona", "📊 Feature Importance", "🔬 Metriche"])
        
        with tab1:
            st.markdown(f"""
            <div class='info-box'>
            <h3>🌳 Random Forest - 100 Alberi Decisionali</h3>
            
            <p><strong>Come funziona:</strong> Crea 100 alberi indipendenti che "votano" il risultato.</p>
            
            <p><strong>Dataset:</strong> {num_giorni} giorni di allenamenti</p>
            
            <p><strong>Parametri Analizzati:</strong></p>
            <ul>
            <li>📏 Distanza (km) - volume allenamento</li>
            <li>😴 Ore Sonno - capacità di recupero</li>
            <li>🧠 Stress Lavoro - carico mentale (1-10)</li>
            <li>❤️ FC Media - intensità cardiaca (bpm)</li>
            <li>💪 RPE - sforzo percepito (1-10)</li>
            </ul>
            
            <p><strong>Output:</strong> Probabilità di rischio infortunio (0-100%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Importanza Relativa dei Parametri**")
            
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            features_sorted = [x[0] for x in imp_data]
            importances_sorted = [x[1] for x in imp_data]
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=features_sorted,
                x=[x*100 for x in importances_sorted],
                orientation='h',
                marker=dict(color=importances_sorted, colorscale='Blues', showscale=True)
            ))
            fig_imp.update_layout(
                title="Quali Fattori Influenzano Più il Rischio?",
                xaxis_title="Importanza (%)",
                yaxis_title="",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown(f"""
            <div class='success-box'>
            <p><strong>Fattore Più Importante: {features_sorted[0]} ({importances_sorted[0]*100:.1f}%)</strong></p>
            <p>Questo parametro influenza più di tutti gli altri la predizione del modello.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("**Performance Metriche**")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%", "Predizioni corrette")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%", "Falsi allarmi bassi")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%", "Cattura rischi reali")
            
            st.markdown("---")
            
            st.markdown("**Confusion Matrix**")
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predetto: Sicuro', 'Predetto: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues'
            ))
            fig_cm.update_layout(
                title="Come il Modello Predice",
                xaxis_title="Previsione Modello",
                yaxis_title="Realtà Storica",
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            <div class='info-box'>
            <p><strong>{cm[0,0]}</strong> (alto-sx): Giorni SICURI predetti CORRETTAMENTE ✅</p>
            <p><strong>{cm[0,1]}</strong> (alto-dx): Giorni SICURI predetti come RISCHIO (Falsi Allarmi)</p>
            <p><strong>{cm[1,0]}</strong> (basso-sx): Giorni A RISCHIO predetti come SICURI (Pericoloso!)</p>
            <p><strong>{cm[1,1]}</strong> (basso-dx): Giorni A RISCHIO predetti CORRETTAMENTE ✅</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

# =====================================================================
# PAGINA 5: CONSIGLIO FINALE
# =====================================================================
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
            emoji_rec = "✅"
        elif risk_score < 60:
            status_emoji = "🟡"
            title = "RECUPERO ATTIVO CONSIGLIATO"
            color = "#fbbc04"
            emoji_rec = "⚠️"
        else:
            status_emoji = "🔴"
            title = "RIPOSO OBBLIGATORIO"
            color = "#ea4335"
            emoji_rec = "❌"
        
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
        st.subheader("📈 Grafici Trend")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        
        with col_g1:
            df_recent = df.tail(30).copy()
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True,
                              title="Sonno Ultimi 30gg")
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_sonno, use_container_width=True)
            st.caption(f"Oggi: {r['ore_sonno']:.1f}h | Media: {media_sonno_90:.1f}h")
        
        with col_g2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True,
                            title="RPE Ultimi 30gg")
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_rpe, use_container_width=True)
            st.caption(f"Oggi: {r['rpe_previsto']}/10 | Media: {media_rpe_90:.1f}/10")
        
        with col_g3:
            fig_stress = px.line(df_recent, y='Stress Lavoro', height=300, markers=True,
                               title="Stress Ultimi 30gg")
            fig_stress.add_hline(y=r['stress_lavoro'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_stress, use_container_width=True)
            st.caption(f"Oggi: {r['stress_lavoro']}/10 | Media: {media_stress_90:.1f}/10")
        
        st.markdown("---")
        st.subheader("💡 Raccomandazioni")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if risk_score < 25:
                st.markdown("""
                <div class='success-box'>
                <h3>✅ Corpo Pronto</h3>
                <ul>
                <li><strong>Intervalli:</strong> 6-8 × 800m (2' recupero)</li>
                <li><strong>Tempo Run:</strong> 3 × 10min sostenuto</li>
                <li><strong>Ripetute:</strong> 5 × 2km (3' recupero)</li>
                <li><strong>Test Velocità:</strong> Perfetto per progress</li>
                <li><strong>Gara:</strong> Condizioni ideali</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown(f"""
                <div class='warning-box'>
                <h3>🟡 Recupero Attivo</h3>
                <ul>
                <li><strong>Easy Run:</strong> Ritmo conversativo 30-40 min</li>
                <li><strong>Recovery Run:</strong> 5-8 km mobilità</li>
                <li><strong>Cross-Training:</strong> Nuoto/ciclismo leggero</li>
                <li><strong>Long Run Easy:</strong> 12-15 km slow</li>
                </ul>
                <p><strong>Dormi:</strong> {max(8, r['ore_sonno'] + 1):.1f}h stasera</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='danger-box'>
                <h3>🔴 Riposo Totale</h3>
                <ul>
                <li>❌ Non correre</li>
                <li>✓ Camminate max 15 min</li>
                <li>✓ Stretching delicato</li>
                <li>✓ Sonno 9-10h</li>
                <li>✓ Recupero prioritario</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col_rec2:
            if risk_score < 25:
                st.markdown("""
                <div class='info-box'>
                <h3>🔄 Post-Allenamento</h3>
                <ul>
                <li>Entro 30 min: Proteine + carbs</li>
                <li>1-2h: Pasto completo</li>
                <li>Idratazione: 500ml acqua per 500kcal</li>
                <li>Sonno: Dormi 1h prima del solito</li>
                </ul>
                <h3>⏰ Prossimi Giorni</h3>
                <ul>
                <li><strong>Domani:</strong> Easy 6-8 km</li>
                <li><strong>+2:</strong> Recupero attivo</li>
                <li><strong>+3:</strong> Allenamento moderato</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif risk_score < 60:
                st.markdown("""
                <div class='info-box'>
                <h3>💤 Priorità Recupero</h3>
                <ul>
                <li>Sonno: 8-9h prioritario</li>
                <li>Idratazione: 3-4 litri acqua</li>
                <li>Stretching: 20-30 minuti</li>
                <li>Nutrizione: Pasti bilanciati</li>
                <li>Stress: Riduci impegni</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class='info-box'>
                <h3>🚑 Monitora</h3>
                <ul>
                <li>Dolore acuto → Medico</li>
                <li>Febbre > 37.5°C → Medico</li>
                <li>Tachicardia FC > 90 → Medico</li>
                <li>Vertigini → Medico</li>
                </ul>
                <h3>📋 Piano Recovery</h3>
                <ul>
                <li><strong>OGGI:</strong> Riposo totale</li>
                <li><strong>+1-2:</strong> Camminate 15 min</li>
                <li><strong>+3:</strong> Easy 20 min</li>
                <li><strong>+4+:</strong> Valuta ritorno</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Riepilogo KPI")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'FC Riposo', 'Recovery', 'SMA', 'Risk'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10",
                      f"{r['fc_riposo']} bpm", f"{recovery_score:.0f}%", f"{sma:.1f}", f"{risk_score:.0f}%"],
            'Stato': [
                "✅ OK" if r['ore_sonno'] >= 7 else "⚠️ Basso",
                "✅ OK" if r['stress_lavoro'] <= 5 else "⚠️ Alto",
                "✅ OK" if r['rpe_previsto'] <= 5 else "⚠️ Intenso",
                "✅ OK" if r['fc_riposo'] <= 65 else "⚠️ Elevata",
                "✅ OK" if recovery_score >= 75 else "⚠️ Moderato" if recovery_score >= 40 else "❌ Pessimo",
                "✅ OK" if sma < 10 else "⚠️ Moderato" if sma < 15 else "❌ Squilibrato",
                "✅ Sicuro" if risk_score < 25 else "⚠️ Moderato" if risk_score < 60 else "🔴 CRITICO"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
