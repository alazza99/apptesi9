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

if pagina == "📋 Analisi Completa":
    st.title("📋 Analisi Completa")
    
    with st.form("form_analisi"):
        st.markdown("### 🎯 Obiettivi")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Obiettivo Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            obj_lt = st.text_input("Obiettivo Lungo Termine", placeholder="Es: Maratona < 3:30")
        
        st.markdown("---")
        
        st.markdown("### 😴 Sonno")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno", 2.0, 12.0, 7.5)
        with col_s2:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
        
        st.markdown("---")
        
        st.markdown("### 🧠 Stress")
        stress_lavoro = st.slider("Stress Lavoro (1-10)", 1, 10, 5)
        
        st.markdown("---")
        
        st.markdown("### ⚡ Allenamento")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            tipo_allenamento = st.selectbox("Tipo", ["Easy", "Long", "Fartlek", "Intervalli", "Tempo", "Gara"])
        with col_a2:
            rpe_previsto = st.slider("RPE (1-10)", 1, 10, 6)
        
        if st.form_submit_button("🚀 ANALIZZA", use_container_width=True):
            st.session_state.analisi_fatta = True
            st.session_state.risultati_analisi = {
                'obj_oggi': obj_oggi,
                'obj_lt': obj_lt,
                'ore_sonno': ore_sonno,
                'fc_riposo': fc_riposo,
                'stress_lavoro': stress_lavoro,
                'tipo_allenamento': tipo_allenamento,
                'rpe_previsto': rpe_previsto,
            }
            st.success("✓ Analisi completata!")

elif pagina == "📈 Statistiche":
    st.title("📈 Statistiche - 90 Giorni")
    
    df = st.session_state.dati.copy()
    
    st.subheader("📊 Panoramica")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("🏃 KM Totali", f"{df['Distanza (km)'].sum():.0f} km")
    col_m2.metric("📊 Sessioni", f"{len(df)}")
    col_m3.metric("📐 Media", f"{df['Distanza (km)'].mean():.1f} km")
    col_m4.metric("⚠️ Giorni Rischio", f"{df['Rischio Infortunio'].sum()}")
    
    st.markdown("---")
    
    st.subheader("📖 Analisi Dettagliata")
    
    tab1, tab2, tab3 = st.tabs(["Volume", "Intensità", "Recupero"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distanza per Settimana**")
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', height=350, color_discrete_sequence=['#1a73e8'])
            fig1.update_layout(xaxis_title="Settimana", yaxis_title="KM")
            st.plotly_chart(fig1, use_container_width=True)
            st.info("Trend volume di allenamento. Incremento graduale è ideale (10% settimanale).")
        
        with col2:
            st.markdown("**Distanza Cumulativa**")
            df['Cumulativa'] = df['Distanza (km)'].cumsum()
            fig_cum = px.line(df, x='Giorno', y='Cumulativa', height=350, markers=True)
            fig_cum.update_layout(xaxis_title="Data", yaxis_title="KM Cumulativi")
            st.plotly_chart(fig_cum, use_container_width=True)
            st.info(f"Totale: {df['Cumulativa'].iloc[-1]:.0f} km. Ogni punto è il totale fino a quel giorno.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FC Media vs Velocità**")
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', size='Distanza (km)',
                            color='RPE', color_continuous_scale='Blues', height=350, opacity=0.7)
            fig2.update_layout(xaxis_title="Velocità (km/h)", yaxis_title="FC Media (bpm)")
            st.plotly_chart(fig2, use_container_width=True)
            st.info("Punti bassi = migliore efficienza cardiaca. Il colore indica l'intensità (RPE).")
        
        with col2:
            st.markdown("**Distribuzione RPE**")
            fig3 = px.histogram(df, x='RPE', nbins=9, height=350, color_discrete_sequence=['#1a73e8'])
            fig3.update_layout(xaxis_title="RPE (1-10)", yaxis_title="Giorni")
            st.plotly_chart(fig3, use_container_width=True)
            st.info("Modello Polarized: 80% facile (RPE 1-3) + 20% intenso (RPE 7-10). Evita il 40-60%.")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sonno Giornaliero**")
            fig_sleep = px.line(df, x='Giorno', y='Ore Sonno', height=350, markers=True)
            fig_sleep.add_hline(y=7.5, line_dash="dash", line_color="green", annotation_text="Target")
            fig_sleep.add_hline(y=6.5, line_dash="dash", line_color="red", annotation_text="Minimo")
            fig_sleep.update_layout(xaxis_title="Data", yaxis_title="Ore di Sonno")
            st.plotly_chart(fig_sleep, use_container_width=True)
            st.info("Target 7.5h. Sotto 6.5h aumenta rischio infortunio.")
        
        with col2:
            st.markdown("**Sonno vs RPE**")
            fig4 = px.scatter(df, x='Ore Sonno', y='RPE', size='Distanza (km)',
                            color='Rischio Infortunio', color_continuous_scale=['lightblue', 'red'],
                            height=350, opacity=0.8)
            fig4.add_hline(y=7, line_dash="dash", line_color="orange")
            fig4.add_vline(x=6.5, line_dash="dash", line_color="orange")
            fig4.update_layout(xaxis_title="Ore Sonno", yaxis_title="RPE")
            st.plotly_chart(fig4, use_container_width=True)
            st.info("Punti rossi = giorni a rischio. Zone: RPE>7 + Sonno<6.5 = PERICOLO.")

elif pagina == "📊 KPI Dashboard":
    st.title("📊 Dashboard KPI")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima.")
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
            <div style='background: {status_color}15; border: 4px solid {status_color}; border-radius: 20px; padding: 40px; text-align: center;'>
            <p style='font-size: 3em; margin: 0; color: {status_color};'>{status_emoji}</p>
            <p style='font-size: 2.8em; margin: 15px 0; font-weight: bold; color: {status_color};'>{risk_score:.0f}%</p>
            <p style='font-size: 1.2em; color: #666; margin: 10px 0;'><strong>Rischio Infortunio</strong></p>
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
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            df_recent = df.tail(30).copy()
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title="Risk Score %",
                gauge={'axis': {'range': [0, 100]},
                        'bar': {'color': status_color},
                        'steps': [
                            {'range': [0, 25], 'color': "#e6f4ea"},
                            {'range': [25, 60], 'color': "#fef7e0"},
                            {'range': [60, 100], 'color': "#fce8e6"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_g2:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[r['ore_sonno'], r['stress_lavoro'], r['rpe_previsto'], recovery_score/20],
                theta=['Sonno (h)', 'Stress', 'RPE', 'Recovery'],
                fill='toself',
                name='Odierno'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                height=350,
                title="Panoramica Parametri"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("😴 Sonno", f"{r['ore_sonno']:.1f}h")
        col_p2.metric("🧠 Stress", f"{r['stress_lavoro']}/10")
        col_p3.metric("⚡ RPE", f"{r['rpe_previsto']}/10")
        col_p4.metric("❤️ FC Riposo", f"{r['fc_riposo']} bpm")

elif pagina == "🔮 ML Explained":
    st.title("🔮 Machine Learning")
    
    try:
        df = st.session_state.dati.copy()
        num_giorni = len(df)
        
        X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
        y_train = df['Rischio Infortunio'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf_model.fit(X_scaled, y_train)
        
        y_pred = rf_model.predict(X_scaled)
        
        acc = accuracy_score(y_train, y_pred)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        cm = confusion_matrix(y_train, y_pred)
        
        feature_names = ['Distanza', 'Sonno', 'Stress', 'FC', 'RPE']
        importances = rf_model.feature_importances_
        
        tab1, tab2, tab3 = st.tabs(["🌳 Modello", "📊 Features", "🔬 Metriche"])
        
        with tab1:
            st.markdown("""
            <div class='info-box'>
            <h3>Random Forest - 100 Alberi</h3>
            <p>Analizza 90 giorni di allenamenti per predire il rischio infortunio.</p>
            <p><strong>Come:</strong> Crea 100 alberi indipendenti che votano il risultato.</p>
            <p><strong>Parametri:</strong> Distanza, Sonno, Stress, FC Media, RPE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            imp_data = list(zip(feature_names, importances))
            imp_data.sort(key=lambda x: x[1], reverse=True)
            
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=[x[0] for x in imp_data],
                x=[x[1] for x in imp_data],
                orientation='h',
                marker=dict(color='#1a73e8')
            ))
            fig_imp.update_layout(
                title="Importanza Parametri",
                xaxis_title="Importanza",
                yaxis_title="",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with tab3:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("✅ Accuracy", f"{acc*100:.1f}%")
            col_m2.metric("🎯 Precision", f"{prec*100:.1f}%")
            col_m3.metric("🔍 Recall", f"{rec*100:.1f}%")
            
            st.markdown("---")
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Pred: Sicuro', 'Pred: Rischio'],
                y=['Reale: Sicuro', 'Reale: Rischio'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 18},
                colorscale='Blues'
            ))
            fig_cm.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
    
    except Exception as e:
        st.error(f"Errore ML: {str(e)}")

elif pagina == "💡 Consiglio Finale":
    st.title("💡 Consiglio Personalizzato")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Completa il questionario prima.")
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
            status_emoji = "🟢"
            title = "ALLENAMENTO INTENSO"
            color = "#34a853"
        elif risk_score < 60:
            status_emoji = "🟡"
            title = "RECUPERO ATTIVO"
            color = "#fbbc04"
        else:
            status_emoji = "🔴"
            title = "RIPOSO OBBLIGATORIO"
            color = "#ea4335"
        
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{status_emoji} {title}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            df_recent = df.tail(30).copy()
            fig_sonno = px.line(df_recent, y='Ore Sonno', height=300, markers=True)
            fig_sonno.add_hline(y=r['ore_sonno'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_sonno, use_container_width=True)
        
        with col_t2:
            fig_rpe = px.line(df_recent, y='RPE', height=300, markers=True)
            fig_rpe.add_hline(y=r['rpe_previsto'], line_dash="dash", line_color="red")
            st.plotly_chart(fig_rpe, use_container_width=True)
        
        st.markdown("---")
        
        if risk_score < 25:
            st.markdown("""
            <div class='success-box'>
            <h3>✅ Corpo Pronto</h3>
            <ul>
            <li><strong>Intervalli:</strong> 6-8 × 800m (2' recupero)</li>
            <li><strong>Tempo Run:</strong> 3 × 10min sostenuto</li>
            <li><strong>Test Velocità:</strong> Perfetto</li>
            <li><strong>Gara:</strong> Condizioni ideali</li>
            </ul>
            <p><strong>Domani:</strong> Easy run 6-8 km</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif risk_score < 60:
            st.markdown(f"""
            <div class='warning-box'>
            <h3>🟡 Recupero Prioritario</h3>
            <ul>
            <li><strong>Easy Run:</strong> Ritmo conversativo 30-40 km</li>
            <li><strong>Recovery Run:</strong> 5-8 km mobilità</li>
            <li><strong>Cross-Training:</strong> Nuoto/ciclismo leggero</li>
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
            </ul>
            <p><strong>Priorità:</strong> Recupero, nutrizione, sonno</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        riepilogo_df = pd.DataFrame({
            'Parametro': ['Sonno', 'Stress', 'RPE', 'Recovery', 'Risk', 'SMA'],
            'Valore': [f"{r['ore_sonno']:.1f}h", f"{r['stress_lavoro']}/10", f"{r['rpe_previsto']}/10", 
                      f"{recovery_score:.0f}%", f"{risk_score:.0f}%", f"{sma:.1f}"],
            'Stato': [
                "✅" if r['ore_sonno'] >= 7 else "⚠️",
                "✅" if r['stress_lavoro'] <= 5 else "⚠️",
                "✅" if r['rpe_previsto'] <= 5 else "⚠️",
                "✅" if recovery_score >= 75 else "⚠️",
                "✅" if risk_score < 25 else "⚠️" if risk_score < 60 else "🔴",
                "✅" if sma < 10 else "⚠️"
            ]
        })
        
        st.dataframe(riepilogo_df, use_container_width=True, hide_index=True)
