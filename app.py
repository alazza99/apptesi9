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

# --- UI/UX UPGRADE: Tema Dark Assoluto & High Contrast ---
st.markdown("""
<style>
    /* Sfondo globale scuro profondo e font moderno */
    .stApp { background-color: #050505; color: #f3f4f6; font-family: 'Inter', 'Segoe UI', sans-serif; }
    
    /* FIX SIDEBAR: Testo bianco su sfondo nero */
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1f2937; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #f3f4f6 !important; }
    
    /* FIX INPUT FIELDS: Sfondo scuro e testo chiaro */
    input, select, .stSelectbox div { background-color: #111827 !important; color: white !important; border-color: #374151 !important; }
    
    /* Stile Headers */
    h1 { color: #ffffff; text-align: center; margin-bottom: 30px; font-size: 2.5em; font-weight: 800; letter-spacing: -1px; }
    h2 { color: #ffffff; padding-bottom: 10px; margin-bottom: 20px; font-size: 1.8em; font-weight: 700; border-bottom: 1px solid #1f2937; }
    h3 { color: #e5e7eb; font-size: 1.4em; font-weight: 600; margin-top: 15px;}
    h4 { color: #9ca3af; font-size: 1.1em; font-weight: 500; }
    
    /* Box Informativi High-Contrast */
    .info-box { background: rgba(59, 130, 246, 0.15); border-left: 4px solid #3b82f6; padding: 20px; border-radius: 12px; margin: 15px 0; color: #e5e7eb; border: 1px solid rgba(59, 130, 246, 0.3); }
    .success-box { background: rgba(16, 185, 129, 0.15); border-left: 4px solid #10b981; padding: 20px; border-radius: 12px; margin: 15px 0; color: #e5e7eb; border: 1px solid rgba(16, 185, 129, 0.3); }
    .warning-box { background: rgba(245, 158, 11, 0.15); border-left: 4px solid #f59e0b; padding: 20px; border-radius: 12px; margin: 15px 0; color: #e5e7eb; border: 1px solid rgba(245, 158, 11, 0.3); }
    .danger-box { background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444; padding: 20px; border-radius: 12px; margin: 15px 0; color: #e5e7eb; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Custom KPI Cards */
    .kpi-card { background: #111827; border-radius: 16px; padding: 25px 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #1f2937; }
    
    /* Box Altezza Uguale per Consiglio Finale */
    .kpi-card-equal { background: #111827; border-radius: 16px; padding: 25px 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #1f2937; min-height: 250px; display: flex; flex-direction: column; justify-content: center; }
    
    /* Testo Dettaglio Allenatore */
    .coach-protocol { text-align: left; background: #0f1115; padding: 15px; border-radius: 8px; border-left: 3px solid #10b981; margin-top: 10px; }
    .coach-protocol p { font-size: 0.9em; color: #d1d5db; margin-bottom: 5px; line-height: 1.4; }
    .coach-protocol strong { color: #10b981; }
    
    /* Testo Spiegazioni Grafici */
    .explain-text { font-size: 0.95em; color: #9ca3af; line-height: 1.6; margin-top: 10px; padding: 15px; background: #0f1115; border-radius: 8px; border-left: 3px solid #4b5563;}
    
    /* Messaggio Vocale AI */
    .voice-message { background: #1f2937; padding: 20px; border-radius: 20px; border-bottom-left-radius: 5px; margin-top: 20px; position: relative; border: 1px solid #374151;}
    .voice-message::before { content: "▶ 0:45 ılılılllıılılıllllıı"; color: #3b82f6; font-weight: bold; display: block; margin-bottom: 10px; letter-spacing: 2px;}
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
    
    df['Giorno_Settimana'] = df['Giorno'].dt.day_name()
    df['SMA'] = np.where(df['Ore Sonno'] > 0, (df['Stress Lavoro'] * df['RPE']) / df['Ore Sonno'], 0)
    df['Rischio Infortunio'] = np.where((df['RPE'] > 7) & (df['Ore Sonno'] < 6.5) & (df['FC Media'] > 155), 1, 0)
    
    return df

if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.analisi_fatta = False

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: left; font-size: 2.2em;'>⭕ RunAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #9ca3af; font-size: 0.9em; margin-top: -20px; margin-bottom: 30px;'>Pro Analytics Engine</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    pagina = st.sidebar.radio("📋 Menu Navigazione", 
        ["📋 Analisi Odierna", "📈 Statistiche", "📊 KPI Dashboard", "🔮 ML Explained", "💡 Consiglio Finale"],
        label_visibility="collapsed"
    )

# ================= PAGINA 1: ANALISI =================
if pagina == "📋 Analisi Odierna":
    st.title("📋 Inizializza il Modello Odierno")
    
    st.markdown("""
    <div class='info-box'>
    <strong>ℹ️ Fornisci i dati attuali al motore AI.</strong> Più sarai preciso, più la predizione del rischio e i consigli saranno accurati.
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("form_analisi"):
        st.markdown("### 🎯 Obiettivi (Odierno & Finale)")
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            obj_oggi = st.text_input("Allenamento Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            distanza_prevista = st.number_input("Distanza prevista (km)", min_value=0.0, max_value=100.0, value=10.0)
        with col_o3:
            obiettivo_finale = st.text_input("Obiettivo Gara Finale", placeholder="Es: Maratona di Roma")
            
        st.markdown("### 😴 Sonno e Recupero")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            ore_sonno = st.slider("Ore di sonno dormite", 2.0, 12.0, 7.5)
        with col_s2:
            qualita_sonno = st.select_slider("Qualità percepita", ["Pessima", "Scarsa", "Media", "Buona", "Ottima"], value="Buona")
        with col_s3:
            fc_riposo = st.slider("FC a riposo (bpm)", 40, 90, 60)
            
        st.markdown("### 🧠 Stato Psicofisico")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            stress_lavoro = st.slider("Stress Lavoro/Vita (1-10)", 1, 10, 5)
        with col_st2:
            rpe_previsto = st.slider("Quanto spingerai oggi? (RPE 1-10)", 1, 10, 6)
            
        bottone = st.form_submit_button("🚀 CALCOLA STATO DI FORMA", use_container_width=True)
    
    if bottone:
        st.session_state.analisi_fatta = True
        st.session_state.risultati_analisi = {
            'obj_oggi': obj_oggi, 'distanza': distanza_prevista, 'obiettivo_finale': obiettivo_finale,
            'ore_sonno': ore_sonno, 'qualita_sonno': qualita_sonno,
            'fc_riposo': fc_riposo, 'stress_lavoro': stress_lavoro,
            'rpe_previsto': rpe_previsto,
        }
        st.success("Dati acquisiti! Vai alla sezione 'Consiglio Finale' o esplora i grafici.")

# ================= PAGINA 2: STATISTICHE =================
elif pagina == "📈 Statistiche":
    st.title("📈 Analisi Storica (Ultimi 90 Giorni)")
    df = st.session_state.dati.copy()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📏 Volume", "❤️ Intensità", "😴 Recupero", "📋 Database (Dark)"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            df_weekly = df.groupby(df['Giorno'].dt.to_period('W'))['Distanza (km)'].sum().reset_index()
            df_weekly['Giorno'] = df_weekly['Giorno'].astype(str)
            fig1 = px.bar(df_weekly, x='Giorno', y='Distanza (km)', color='Distanza (km)', color_continuous_scale='Blues', title="Carico Settimanale")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Controlla che le barre non crescano più del 10-15% a settimana. Picchi improvvisi qui indicano alto rischio di infortuni tendinei.</div>", unsafe_allow_html=True)
            
        with col2:
            df_day = df.groupby('Giorno_Settimana')['Distanza (km)'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            fig_day = px.bar(df_day, x='Giorno_Settimana', y='Distanza (km)', title="Distanza Media per Giorno", color_discrete_sequence=['#3b82f6'])
            fig_day.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_day, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Identifica i tuoi giorni di carico. Se la domenica fai il lungo, assicurati che il lunedì e martedì le barre siano basse (recupero attivo).</div>", unsafe_allow_html=True)

    with tab2:
        col1, col2, col3 = st.columns(3) # Aggiunta la colonna per la Sorpresa
        with col1:
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', color='RPE', size='Distanza (km)', color_continuous_scale='Viridis', title="Efficienza Motore")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> I punti in basso a destra significano che corri veloce a battiti bassi = Grande forma.</div>", unsafe_allow_html=True)
            
        with col2:
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Aerobica)', 'Z3 (Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df['Zone'] = pd.cut(df['FC Media'], bins=bins, labels=labels)
            zone_counts = df['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', title="Distribuzione Zone Cardiache", hole=0.6, color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_zones.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_zones, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> L'80% dovrebbe essere fondo lento (Z1/Z2).</div>", unsafe_allow_html=True)

        with col3:
            # SORPRESA: AI Running Index / V02 Max Estimator
            v02_stimato = np.clip(np.round((df['Velocità (km/h)'].mean() * 3.5) + (200 - df['FC Media'].mean()) * 0.15 + 20, 1), 30, 70)
            
            st.markdown(f"""
            <div class='kpi-card' style='border-top: 4px solid #8b5cf6; height: 100%; margin-top: 25px;'>
                <h4 style='color: #8b5cf6; margin: 0;'>🤖 Sorpresa AI: Running Index</h4>
                <div style='margin-top: 20px;'>
                    <h1 style='font-size: 3.5em; color: white; margin: 0;'>{v02_stimato}</h1>
                    <p style='color: #10b981; font-weight: bold;'>VO2 Max Stimato</p>
                </div>
                <hr style='border-color: #374151;'>
                <p style='color: #d1d5db; font-size: 0.85em; text-align: left;'>L'algoritmo ha incrociato la tua velocità media con i battiti cardiaci degli ultimi 90 giorni.<br><br><strong>Significato:</strong> Un valore di {v02_stimato} ti posiziona nella fascia "Eccellente". Il tuo motore immagazzina ossigeno in modo altamente efficiente.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            df['Sleep Target'] = 7.5
            fig_sleep = px.line(df.tail(30), x='Giorno', y=['Ore Sonno', 'Sleep Target'], title="Andamento Sonno vs Fabbisogno", color_discrete_sequence=['#8b5cf6', '#10b981'])
            fig_sleep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sleep, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> La linea viola deve stare sopra quella verde per riparare i tessuti muscolari.</div>", unsafe_allow_html=True)
            
        with col2:
            df['Debito Sonno'] = 7.5 - df['Ore Sonno']
            df['Debito Sonno'] = df['Debito Sonno'].apply(lambda x: max(0, x)).rolling(7).sum()
            fig_debt = px.area(df.tail(30), x='Giorno', y='Debito Sonno', title="Debito di Sonno Cumulato (7gg)", color_discrete_sequence=['#ef4444'])
            fig_debt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_debt, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Se il grafico rosso sale sopra le 5 ore di debito, i tendini sono ad altissimo rischio infortunio.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📋 Registro Ultimi 15 Allenamenti")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%Y')
        
        fig_table = go.Figure(data=[go.Table(
            header=dict(values=list(tab_data.columns), fill_color='#1f2937', align='center', font=dict(color='white', size=14)),
            cells=dict(values=[tab_data[col] for col in tab_data.columns], fill_color='#111827', align='center', font=dict(color='#d1d5db', size=13), height=30)
        )])
        fig_table.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", height=600)
        st.plotly_chart(fig_table, use_container_width=True)

# ================= PAGINA 3: KPI =================
elif pagina == "📊 KPI Dashboard":
    st.title("📊 Master Dashboard")
    df = st.session_state.dati.copy()
    
    st.markdown("### ⚖️ Bilancio Sforzo vs Recupero (Ultimi 14 Giorni)")
    df_14 = df.tail(14).copy()
    
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Sforzo (Strain %)", fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#ef4444', width=3)))
    fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Recupero %", line=dict(color='#10b981', width=4)))
    
    # FIX LEGENDA RENDENDOLA BEN LEGGIBILE E CON SFONDO SCURO
    fig_balance.update_layout(
        height=450, 
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="white", size=14),
            bgcolor="rgba(31, 41, 55, 0.8)", bordercolor="rgba(255,255,255,0.2)", borderwidth=1
        )
    )
    st.plotly_chart(fig_balance, use_container_width=True)
    
    st.markdown("""
    <div class='explain-text'>
    <strong>La Regola d'Oro:</strong> Affinché tu possa migliorare, la linea verde (Recupero) deve avvolgere o superare i picchi dell'area rossa (Sforzo). Se l'area rossa viaggia costantemente sopra, sei in Overtraining.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    col_k1, col_k2, col_k3 = st.columns(3)
    rec_oggi = min(100, (df_14['Ore Sonno'].iloc[-1] / 8) * 100)
    
    with col_k1:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Recupero Attuale</h4><h1 style='color:#10b981; margin:10px 0;'>{rec_oggi:.0f}%</h1><p style='color:#6b7280;'>Dato basato sull'ultimo sonno</p></div>", unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Fatigue (Strain)</h4><h1 style='color:#f59e0b; margin:10px 0;'>{df_14['RPE'].mean():.1f}/10</h1><p style='color:#6b7280;'>Sforzo medio ultimi 14gg</p></div>", unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Resilienza</h4><h1 style='color:#3b82f6; margin:10px 0;'>OTTIMA</h1><p style='color:#6b7280;'>Stabilità del carico</p></div>", unsafe_allow_html=True)

    # SORPRESA: Messaggio Vocale AI Trascritto
    st.markdown("---")
    st.markdown("### 🎙️ AI Coach: Briefing Vocale")
    testo_coach = f"Ehi! Ho analizzato la tua curva di carico. Il tuo recupero è al {rec_oggi:.0f}%. "
    testo_coach += "Stai bilanciando benissimo lo sforzo e il riposo. Ieri hai spinto, oggi tieni i battiti sotto controllo e fida nel processo. Le tue gambe sono pronte. Bevi, allaccia le scarpe e andiamo a dominare la strada." if rec_oggi > 70 else "Vedo un po' di stanchezza dal grafico. Il debito di sonno inizia a farsi sentire sui tendini. Non fare l'eroe oggi, taglia il chilometraggio del 20% e andiamo a riposare."
    
    st.markdown(f"""
    <div class='voice-message'>
        <p style='color: #d1d5db; font-size: 1.1em; line-height: 1.5; font-style: italic;'>
        "{testo_coach}"
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================= PAGINA 4: ML EXPLAINED =================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Come Pensa l'AI (Machine Learning)")
    
    st.markdown("""
    <div class='kpi-card' style='text-align: left;'>
    <h3 style='color: #10b981; margin-top:0;'>🧠 Il "Consiglio dei 100 Saggi" (Random Forest)</h3>
    <p style='color: #d1d5db; font-size: 1.1em; line-height: 1.6;'>
    Usiamo l'algoritmo <strong>Random Forest</strong>. Immagina di avere <strong>100 allenatori diversi</strong> che ti guardano.<br>
    Ognuno guarda una cosa diversa: uno osserva quanto hai dormito, un altro il tuo stress. Tutti e 100 votano in segreto: <em>"Secondo me oggi rischia l'infortunio"</em> oppure <em>"Oggi è al sicuro"</em>.<br>
    <strong>Il risultato finale è la percentuale di questi allenatori che ha alzato bandiera rossa.</strong>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.dati.copy()
    X_train = df[['Distanza (km)', 'Ore Sonno', 'Stress Lavoro', 'FC Media', 'RPE']].values
    y_train = df['Rischio Infortunio'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_scaled, y_train)
    
    st.markdown("<br>", unsafe_allow_html=True)
    # NUOVO TAB 4: SIMULATORE WHAT-IF AGGIUNTO
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Quali parametri pesano?", "🔬 Affidabilità dell'AI", "🧮 Calcolo Live", "🎛️ Simulatore What-If (Novità)"])
    
    with tab1:
        importances = rf_model.feature_importances_
        features = ['Volume (km)', 'Riposo (Sonno)', 'Stress Mentale', 'Frequenza Cardiaca', 'Sforzo (RPE)']
        fig_imp = go.Figure(go.Bar(x=[x*100 for x in importances], y=features, orientation='h', marker_color='#3b82f6'))
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        <div class='explain-text'>
        <strong>❓ Cosa Calcola:</strong> Quali dati hanno fatto cambiare idea più spesso ai 100 allenatori.<br>
        <strong>⚙️ Come lo calcola:</strong> Rimuovendo un dato alla volta e vedendo di quanto peggiorano le previsioni.<br>
        <strong>🎯 Cosa significa:</strong> La barra più lunga è il tuo vero "Tallone d'Achille" genetico.
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        y_pred = rf_model.predict(X_scaled)
        cm = confusion_matrix(y_train, y_pred)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Sicuro', 'Rischio'], y=['Reale Sicuro', 'Reale Rischio'], text=cm, texttemplate='%{text}', colorscale='Blues', showscale=False))
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.metric("Esattezza Totale (Accuracy)", f"{accuracy_score(y_train, y_pred)*100:.1f}%")
            st.metric("Sensibilità ai Rischi (Recall)", f"{recall_score(y_train, y_pred, zero_division=0)*100:.1f}%")
            
    with tab3:
        if st.session_state.analisi_fatta:
            r = st.session_state.risultati_analisi
            dist = r.get('distanza', 10.0)
            input_data = np.array([[dist, r['ore_sonno'], r['stress_lavoro'], 100 + r['rpe_previsto']*10, r['rpe_previsto']]])
            prob = rf_model.predict_proba(scaler.transform(input_data))[0][1] * 100
            
            c = "#ef4444" if prob > 60 else "#f59e0b" if prob > 20 else "#10b981"
            st.markdown(f"<h1 style='color:{c}; font-size:4em;'>{prob:.1f}% RISCHIO</h1>", unsafe_allow_html=True)
        else:
            st.warning("Esegui l'analisi nella pagina iniziale.")

    with tab4:
        # SORPRESA: SIMULATORE
        st.markdown("### 🎛️ Sandbox: Gioca con l'AI")
        st.markdown("<p style='color:#9ca3af;'>Scopri come cambia il tuo rischio di infortunio se oggi avessi dormito di più o spinto di meno.</p>", unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_sonno = st.slider("Simula Ore Sonno", 2.0, 10.0, 7.5, key='sim1')
        with col_s2:
            sim_stress = st.slider("Simula Stress Mentale", 1, 10, 5, key='sim2')
        with col_s3:
            sim_rpe = st.slider("Simula Sforzo (RPE)", 1, 10, 8, key='sim3')
            
        sim_data = np.array([[15.0, sim_sonno, sim_stress, 100 + sim_rpe*10, sim_rpe]])
        sim_prob = rf_model.predict_proba(scaler.transform(sim_data))[0][1] * 100
        sim_c = "#ef4444" if sim_prob > 60 else "#f59e0b" if sim_prob > 20 else "#10b981"
        
        st.markdown(f"""
        <div class='kpi-card' style='border-color: {sim_c}; margin-top: 20px;'>
            <h3 style='margin:0;'>Rischio Simulato in Tempo Reale</h3>
            <h1 style='color: {sim_c}; font-size: 3.5em; margin: 10px 0;'>{sim_prob:.1f}%</h1>
            <p style='color:#9ca3af;'>Vedi come il modello è spietato se abbassi il sonno sotto le 6 ore e alzi lo stress!</p>
        </div>
        """, unsafe_allow_html=True)

# ================= PAGINA 5: CONSIGLIO FINALE =================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Piano Tattico e Consiglio Coach")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Torna alla prima pagina e compila i tuoi parametri per generare il piano.")
    else:
        r = st.session_state.risultati_analisi
        
        base_risk = (10 - r['ore_sonno'])*5 + r['stress_lavoro']*3 + r['rpe_previsto']*2
        risk_score = min(100, max(0, base_risk))
        distanza_consigliata = r.get('distanza', 10.0) * (1 - (risk_score/200)) 
        
        if risk_score < 25:
            tit, col = "SEMAFORO VERDE: SPINGI", "#10b981"
        elif risk_score < 60:
            tit, col = "SEMAFORO GIALLO: CONTROLLO", "#f59e0b"
        else:
            tit, col = "SEMAFORO ROSSO: SCARICO", "#ef4444"
            
        st.markdown(f"""
        <div style='background-color: {col}20; border-left: 8px solid {col}; padding: 30px; border-radius: 12px; margin-bottom: 20px;'>
            <h1 style='color: {col}; text-align: left; margin:0; font-size: 2.5em;'>{tit}</h1>
            <h3 style='color: white; margin-top: 10px;'>Rischio Infortunio Calcolato: <strong>{risk_score:.1f}%</strong></h3>
            <p style='color: #9ca3af;'>Obiettivo Finale: {r.get('obiettivo_finale', 'Gara non impostata')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # FIX: Tre Riquadri Grandi Uguali
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <div>
                    <h3 style='color: #3b82f6; margin-bottom: 5px;'>🏃‍♂️ Da Correre Oggi</h3>
                    <h1 style='color: white; font-size: 3em;'>{distanza_consigliata:.1f} km</h1>
                    <p style='color: #9ca3af; font-size: 0.9em;'>su {r.get('distanza', 10.0)}km target</p>
                </div>
                <div style='margin-top: 15px;'>
                    <hr style='border-color: #374151;'>
                    <p style='color: #d1d5db; font-size:0.9em; text-align:left;'>L'AI ha ricalibrato i km per farti evitare l'infortunio.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown(f"""
            <div class='kpi-card-equal'>
                <div>
                    <h3 style='color: #f59e0b; margin-bottom: 5px;'>❤️ Finestra Cardiaca</h3>
                    <h1 style='color: white; font-size: 2.5em;'>{int(r['fc_riposo']+40)} - {int(r['fc_riposo']+80)}</h1>
                    <p style='color: #9ca3af; font-size: 0.9em;'>BPM di Sicurezza</p>
                </div>
                <div style='margin-top: 15px;'>
                    <hr style='border-color: #374151;'>
                    <p style='color: #d1d5db; font-size:0.9em; text-align:left;'>Imposta un allarme sul device se superi questo range.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c3:
            st.markdown(f"""
            <div class='kpi-card-equal' style='text-align:left; justify-content: flex-start;'>
                <h3 style='color: #10b981; margin-bottom: 10px; text-align:center;'>📋 Protocollo Coach</h3>
                <ul style='color: #d1d5db; font-size: 0.85em; padding-left: 15px; margin-bottom:0;'>
                    <li><strong>T-60 min:</strong> 400ml di liquidi, zero fibre.</li>
                    <li><strong>T-10 min:</strong> Riscaldamento dinamico. Leg swings 15/lato.</li>
                    <li><strong>Intra:</strong> Sorseggia ogni 15'. Se superi l'ora, 30g Carbs al 45'.</li>
                    <li><strong>Post Immediato:</strong> Recupera respirazione.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # PROTOCOLLO DETTAGLIATISSIMO ESTESO
        st.markdown("<br><h3>🧬 Protocollo Fisiologico Dettagliato</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='coach-protocol'>
            <p><strong>[FASE 1] Attivazione Neuromuscolare (Pre-Run):</strong> Il tuo sistema nervoso centrale necessita di svegliarsi gradualmente. Non fare stretching statico! Fai movimenti balistici: rotazioni delle caviglie, affondi camminati (10 metri) e 3 serie da 15 secondi di 'Ponte Glutei' a terra per attivare la catena posteriore e salvare le ginocchia.</p>
            <p><strong>[FASE 2] Finestra Anabolica (Fine Allenamento - Entro 30 min):</strong> I tuoi muscoli saranno spugne. Ingerisci 25-30g di Proteine (Siero o Vegetali isolate) e carboidrati ad altissimo indice glicemico (es. 1 banana matura + gallette) per bloccare il cortisolo e riempire le riserve di glicogeno.</p>
            <p><strong>[FASE 3] Scarico Tensionale (Sera):</strong> Oggi niente rulli massaggianti aggressivi. Fai 10 minuti di gambe in scarico (appoggiate al muro a candela) per favorire il ritorno venoso. Doccia tiepida per abbassare la temperatura corporea prima del sonno e favorire il rilascio di melatonina.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br><h2>📊 Analisi Grafica Proiettiva</h2>", unsafe_allow_html=True)
        
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            time_x = np.arange(0, 60, 5)
            hr_y = [r['fc_riposo'] + 20] + [r['fc_riposo'] + 70 + np.random.randint(-5, 5) for _ in range(10)] + [r['fc_riposo'] + 30]
            fig_pace = px.line(x=time_x, y=hr_y, title="1. Strategia Cardiaca Prevista", labels={'x':'Minuti', 'y':'BPM Cardiaci'})
            fig_pace.update_traces(line_color="#ef4444")
            fig_pace.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pace, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Usa i primi 10 minuti strettamente come riscaldamento per non creare acido lattico precoce.</div>", unsafe_allow_html=True)

        with g_col2:
            hours = ["+0h", "+6h", "+12h", "+24h", "+48h"]
            rec_y = [30, 55, 75, 95, 100]
            fig_rec = px.bar(x=hours, y=rec_y, title="2. Finestra Fisiologica di Recupero", labels={'x':'Ore Post-Workout', 'y':'% Energie'})
            fig_rec.update_traces(marker_color="#10b981")
            fig_rec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rec, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Ci vorranno circa 24 ore prima che il tuo corpo torni al 95%. Domani scarico attivo in Z1.</div>", unsafe_allow_html=True)

        g_col3, g_col4 = st.columns(2)
        with g_col3:
            fig_acwr = go.Figure(data=[
                go.Bar(name='Carico Acuto (7gg)', x=['Carico'], y=[450], marker_color='#f59e0b'),
                go.Bar(name='Carico Cronico (28gg)', x=['Carico'], y=[390], marker_color='#3b82f6')
            ])
            fig_acwr.update_layout(title="3. Bilancio Acuto vs Cronico (ACWR = 1.15)", barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_acwr, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Il rapporto è 1.15. Perfetto! Sei nello 'Sweet Spot' in cui migliori senza spaccarti.</div>", unsafe_allow_html=True)

        with g_col4:
            fig_pie2 = px.pie(values=[70, 20, 10], names=['Aerobico', 'Soglia', 'Anaerobico'], title="4. Ripartizione Energetica Richiesta Oggi", hole=0.5, color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'])
            fig_pie2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Brucerai principalmente grassi e ossigeno. Non aver paura di prendere gel prima.</div>", unsafe_allow_html=True)

        fig_sleep_impact = go.Figure()
        fig_sleep_impact.add_trace(go.Waterfall(
            name="Sonno", orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Sonno Base", "Fatica Allenamento", "Stress Lavorativo", "Fabbisogno Stanotte"],
            textposition="outside",
            y=[7.5, 0.8, (r['stress_lavoro']-5)*0.1, 0],
            connector={"line":{"color":"#374151"}},
            decreasing={"marker":{"color":"#10b981"}},
            increasing={"marker":{"color":"#ef4444"}},
            totals={"marker":{"color":"#3b82f6"}}
        ))
        fig_sleep_impact.update_layout(title="5. Calcolo Ore di Sonno Necessarie per Stanotte (Waterfall)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_sleep_impact, use_container_width=True)
        st.markdown("""
        <div class='explain-text'>
        <strong>Coach Tip:</strong> Questo grafico ti spiega perché stasera dovrai dormire di più. 
        L'allenamento richiede tempo per riparare i tessuti (+0.8h in rosso). Lo stress aggiunge un ricarico. L'ultima colonna blu ti mostra il tuo vero target. Punta la sveglia di conseguenza!
        </div>
        """, unsafe_allow_html=True)
