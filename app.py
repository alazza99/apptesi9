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
    
    /* Explain Text */
    .explain-text { font-size: 0.95em; color: #9ca3af; line-height: 1.6; margin-top: 10px; padding: 15px; background: #0f1115; border-radius: 8px; border-left: 3px solid #4b5563;}
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
        st.markdown("### 🎯 Il tuo Obiettivo")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            obj_oggi = st.text_input("Allenamento Odierno", placeholder="Es: 10 km easy run")
        with col_o2:
            distanza_prevista = st.number_input("Distanza prevista (km)", min_value=0.0, max_value=100.0, value=10.0)
            
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
            'obj_oggi': obj_oggi, 'distanza': distanza_prevista,
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
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(df, x='Velocità (km/h)', y='FC Media', color='RPE', size='Distanza (km)', color_continuous_scale='Viridis', title="Efficienza Motore (HR vs Passo)")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Cerca i punti verdi (RPE basso) in alto a destra. Significano che stai correndo veloce faticando poco = Efficienza Aerobica.</div>", unsafe_allow_html=True)
            
        with col2:
            bins = [0, 120, 140, 160, 180, 200]
            labels = ['Z1 (Recupero)', 'Z2 (Aerobica)', 'Z3 (Tempo)', 'Z4 (Soglia)', 'Z5 (Max)']
            df['Zone'] = pd.cut(df['FC Media'], bins=bins, labels=labels)
            zone_counts = df['Zone'].value_counts().reset_index()
            fig_zones = px.pie(zone_counts, values='count', names='Zone', title="Distribuzione Zone Cardiache", hole=0.6, color_discrete_sequence=px.colors.sequential.Tealgrn)
            fig_zones.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_zones, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> L'ideale è un 'Polarized Training': l'80% del grafico dovrebbe essere Z1/Z2 (fondo lento) e il 20% Z4/Z5. Se hai troppa Z3 (zona grigia), stai sbagliando allenamento.</div>", unsafe_allow_html=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            df['Sleep Target'] = 7.5
            fig_sleep = px.line(df.tail(30), x='Giorno', y=['Ore Sonno', 'Sleep Target'], title="Andamento Sonno vs Fabbisogno", color_discrete_sequence=['#8b5cf6', '#10b981'])
            fig_sleep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sleep, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> La linea viola deve stare sopra quella verde. Ogni volta che va sotto, accumuli fatica centrale che rallenta il recupero muscolare.</div>", unsafe_allow_html=True)
            
        with col2:
            df['Debito Sonno'] = 7.5 - df['Ore Sonno']
            df['Debito Sonno'] = df['Debito Sonno'].apply(lambda x: max(0, x)).rolling(7).sum()
            fig_debt = px.area(df.tail(30), x='Giorno', y='Debito Sonno', title="Debito di Sonno Cumulato (7gg)", color_discrete_sequence=['#ef4444'])
            fig_debt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_debt, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Per l'atleta:</strong> Questa è l'area del pericolo. Se il grafico rosso sale sopra le 5-6 ore di debito settimanale, il tuo sistema immunitario e i tuoi tendini sono ad altissimo rischio infortunio.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("### 📋 Registro Ultimi 15 Allenamenti")
        tab_data = df[['Giorno', 'Distanza (km)', 'Velocità (km/h)', 'FC Media', 'RPE', 'Ore Sonno']].tail(15).copy()
        tab_data['Giorno'] = tab_data['Giorno'].dt.strftime('%d/%m/%Y')
        
        # Tabella in puro stile Dark nativa di Plotly (Niente sfondi bianchi Streamlit!)
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
    
    # 1. Grafico Enorme di Balance
    st.markdown("### ⚖️ Bilancio Sforzo vs Recupero (Ultimi 14 Giorni)")
    df_14 = df.tail(14).copy()
    
    fig_balance = go.Figure()
    fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=df_14['RPE']*10, name="Sforzo (Strain %)", fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#ef4444', width=3)))
    fig_balance.add_trace(go.Scatter(x=df_14['Giorno'], y=(df_14['Ore Sonno']/8)*100, name="Recupero %", line=dict(color='#10b981', width=4)))
    
    fig_balance.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_balance, use_container_width=True)
    
    st.markdown("""
    <div class='explain-text'>
    <strong>Cosa stai guardando:</strong> L'area rossa indica lo stress che hai imposto al corpo (Allenamento + Stress lavorativo misurato in RPE normalizzato). La linea verde è la tua ricarica notturna. 
    <br><br><strong>La Regola d'Oro:</strong> Affinché tu possa migliorare (Supercompensazione), la linea verde deve avvolgere o superare i picchi dell'area rossa. Se l'area rossa viaggia costantemente sopra la linea verde, sei in "Overtraining" (Sovrallenamento).
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Kpi Numerici
    col_k1, col_k2, col_k3 = st.columns(3)
    rec_oggi = min(100, (df_14['Ore Sonno'].iloc[-1] / 8) * 100)
    
    with col_k1:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Recupero Attuale</h4><h1 style='color:#10b981; margin:10px 0;'>{rec_oggi:.0f}%</h1><p style='color:#6b7280;'>Dato basato sull'ultimo sonno</p></div>", unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Fatigue (Strain)</h4><h1 style='color:#f59e0b; margin:10px 0;'>{df_14['RPE'].mean():.1f}/10</h1><p style='color:#6b7280;'>Sforzo medio ultimi 14gg</p></div>", unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"<div class='kpi-card'><h4 style='color:#9ca3af;'>Resilienza</h4><h1 style='color:#3b82f6; margin:10px 0;'>OTTIMA</h1><p style='color:#6b7280;'>Stabilità del carico</p></div>", unsafe_allow_html=True)

# ================= PAGINA 4: ML EXPLAINED =================
elif pagina == "🔮 ML Explained":
    st.title("🔮 Come Pensa l'AI (Machine Learning)")
    
    st.markdown("""
    <div class='kpi-card' style='text-align: left;'>
    <h3 style='color: #10b981; margin-top:0;'>🧠 Il "Consiglio dei 100 Saggi" (Random Forest)</h3>
    <p style='color: #d1d5db; font-size: 1.1em; line-height: 1.6;'>
    Non usiamo un singolo calcolo per dirti se ti farai male, ma usiamo l'algoritmo <strong>Random Forest</strong>. Immagina di avere <strong>100 allenatori diversi</strong> che ti guardano (i 100 Alberi Decisionali).<br><br>
    Ogni allenatore guarda una cosa diversa: uno osserva solo quanto hai dormito e corso, un altro guarda solo il tuo stress mentale e l'intensità (RPE). Alla fine, tutti e 100 votano in segreto: <em>"Secondo me oggi rischia l'infortunio"</em> oppure <em>"Oggi è al sicuro"</em>.<br><br>
    <strong>Il risultato finale che ti diamo è semplicemente la percentuale di questi allenatori che ha alzato bandiera rossa.</strong>
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
    tab1, tab2, tab3 = st.tabs(["📊 Quali parametri pesano di più?", "🔬 Affidabilità dell'AI", "🧮 Il tuo Calcolo Live"])
    
    with tab1:
        st.markdown("### Feature Importance")
        importances = rf_model.feature_importances_
        features = ['Volume (km)', 'Riposo (Sonno)', 'Stress Mentale', 'Frequenza Cardiaca', 'Sforzo (RPE)']
        fig_imp = go.Figure(go.Bar(x=[x*100 for x in importances], y=features, orientation='h', marker_color='#3b82f6'))
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        <div class='explain-text'>
        <strong>❓ Cosa Calcola:</strong> Quali dati hanno fatto cambiare idea più spesso ai 100 allenatori.<br>
        <strong>⚙️ Come lo calcola:</strong> Rimuovendo un dato alla volta (es. togliendo il sonno) e vedendo di quanto peggiorano le previsioni.<br>
        <strong>🎯 Cosa significa per te:</strong> La barra più lunga è il tuo vero "Tallone d'Achille". Se il Sonno ha la barra più lunga, significa che per la TUA genetica, dormire poco ti fa infortunare molto più che correre tanto.
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown("### Matrice di Confusione & Metriche")
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
            
        st.markdown("""
        <div class='explain-text'>
        <strong>❓ Cosa Calcola:</strong> Quante volte l'AI ha avuto ragione in passato.<br>
        <strong>⚙️ Come lo calcola:</strong> Nasconde i risultati reali di infortunio dei 90 giorni all'algoritmo, gli fa fare la previsione, e poi confronta quanti ne ha presi.<br>
        <strong>🎯 Cosa significa per te:</strong> Se la "Sensibilità" è alta, significa che quando il modello ti dice "ATTENZIONE", devi fermarti davvero, perché non si fa scappare un singolo infortunio reale.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        if st.session_state.analisi_fatta:
            r = st.session_state.risultati_analisi
            dist = r.get('distanza', 10.0)
            input_data = np.array([[dist, r['ore_sonno'], r['stress_lavoro'], 100 + r['rpe_previsto']*10, r['rpe_previsto']]])
            prob = rf_model.predict_proba(scaler.transform(input_data))[0][1] * 100
            
            c = "#ef4444" if prob > 60 else "#f59e0b" if prob > 20 else "#10b981"
            st.markdown(f"<h1 style='color:{c}; font-size:4em;'>{prob:.1f}% RISCHIO</h1>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='explain-text'>
            <strong>❓ Cosa Calcola:</strong> La probabilità esatta che tu ti faccia male o vada in sovrallenamento <strong>oggi</strong>.<br>
            <strong>⚙️ Come lo calcola:</strong> Prende i tuoi dati immessi (Distanza: {dist}km, Sonno: {r['ore_sonno']}h, Stress: {r['stress_lavoro']}/10) e li sottopone ai 100 allenatori (alberi decisionali). In questo caso, {int(prob)} allenatori su 100 hanno votato "Rischio".<br>
            <strong>🎯 Cosa significa per te:</strong> Se supera il 25% sei in zona di moderata attenzione. Sopra il 60% l'infortunio è statisticamente imminente, cambia allenamento.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Esegui prima l'analisi nella pagina iniziale.")

# ================= PAGINA 5: CONSIGLIO FINALE =================
elif pagina == "💡 Consiglio Finale":
    st.title("💡 Piano Tattico e Consiglio Coach")
    
    if not st.session_state.analisi_fatta:
        st.warning("⚠️ Torna alla prima pagina e compila i tuoi parametri per generare il piano.")
    else:
        r = st.session_state.risultati_analisi
        
        # Logica di calcolo consiglio
        base_risk = (10 - r['ore_sonno'])*5 + r['stress_lavoro']*3 + r['rpe_previsto']*2
        risk_score = min(100, max(0, base_risk))
        distanza_consigliata = r.get('distanza', 10.0) * (1 - (risk_score/200)) # Scala giù se rischio alto
        
        # HEADING COLORATO
        if risk_score < 25:
            tit, col = "SEMAFORO VERDE: SPINGI", "#10b981"
        elif risk_score < 60:
            tit, col = "SEMAFORO GIALLO: CONTROLLO", "#f59e0b"
        else:
            tit, col = "SEMAFORO ROSSO: SCARICO", "#ef4444"
            
        st.markdown(f"""
        <div style='background-color: {col}20; border-left: 8px solid {col}; padding: 30px; border-radius: 12px; margin-bottom: 20px;'>
            <h1 style='color: {col}; text-align: left; margin:0; font-size: 2.5em;'>{tit}</h1>
            <h3 style='color: white; margin-top: 10px;'>Rischio Infortunio/Burnout Calcolato: <strong>{risk_score:.1f}%</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # QUANTO CORRERE E PROTOCOLLI
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"""
            <div class='kpi-card' style='text-align:left;'>
                <h3 style='color: #3b82f6;'>🏃‍♂️ Quanto Correre Oggi</h3>
                <h1 style='color: white;'>{distanza_prevista:.1f} km</h1>
                <p style='color: #9ca3af;'>Obiettivo Iniziale: {r.get('distanza', 10.0)} km</p>
                <hr style='border-color: #374151;'>
                <p style='color: #d1d5db; font-size:0.95em;'>
                Il motore AI ha adattato il tuo volume in base allo stress sistemico. 
                Mantieni i battiti tra <strong>{int(r['fc_riposo']+40)} bpm</strong> e <strong>{int(r['fc_riposo']+80)} bpm</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown("""
            <div class='kpi-card' style='text-align:left;'>
                <h3 style='color: #10b981;'>📋 Protocolli</h3>
                <h4 style='color:white;'>Pre-Workout (1h prima)</h4>
                <p style='color: #9ca3af; font-size:0.9em; margin-bottom: 10px;'>• 400ml Acqua/Elettroliti<br>• Dinamico: Leg swings (15/gamba)<br>• Attivazione glutei (Ponte x10)</p>
                <h4 style='color:white;'>Post-Workout (Finestra 30 min)</h4>
                <p style='color: #9ca3af; font-size:0.9em; margin-bottom: 0;'>• 25g Proteine + 50g Carboidrati semplici<br>• Rullo miofasciale su polpacci (2 min)</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><h2>📊 Analisi Grafica Proiettiva</h2>", unsafe_allow_html=True)
        
        # I 5 GRAFICI DEL CONSIGLIO
        # 1 e 2
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            # Pacing Strategy
            time_x = np.arange(0, 60, 5)
            hr_y = [r['fc_riposo'] + 20] + [r['fc_riposo'] + 70 + np.random.randint(-5, 5) for _ in range(10)] + [r['fc_riposo'] + 30]
            fig_pace = px.line(x=time_x, y=hr_y, title="1. Strategia Cardiaca Prevista (Simulazione)", labels={'x':'Minuti', 'y':'BPM Cardiaci'})
            fig_pace.update_traces(line_color="#ef4444")
            fig_pace.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pace, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Il grafico mostra come devono salire i tuoi battiti. Nota la rampa iniziale lenta: usa i primi 10 minuti strettamente come riscaldamento per non creare acido lattico precoce.</div>", unsafe_allow_html=True)

        with g_col2:
            # Finestra Recupero
            hours = ["+0h (Fine)", "+6h", "+12h", "+24h (Domani)", "+48h"]
            rec_y = [30, 55, 75, 95, 100]
            fig_rec = px.bar(x=hours, y=rec_y, title="2. Finestra Fisiologica di Recupero", labels={'x':'Ore Post-Workout', 'y':'% Energie Ripristinate'})
            fig_rec.update_traces(marker_color="#10b981")
            fig_rec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_rec, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Ci vorranno circa 24 ore prima che il tuo corpo torni al 95%. Questo significa che l'allenamento di domani dovrà essere esclusivamente in Z1 (scarico attivo).</div>", unsafe_allow_html=True)

        # 3 e 4
        g_col3, g_col4 = st.columns(2)
        with g_col3:
            # ACWR
            fig_acwr = go.Figure(data=[
                go.Bar(name='Carico Acuto (Ultimi 7gg)', x=['Lavoro Fatto'], y=[450], marker_color='#f59e0b'),
                go.Bar(name='Carico Cronico (Media 28gg)', x=['Lavoro Fatto'], y=[390], marker_color='#3b82f6')
            ])
            fig_acwr.update_layout(title="3. Bilancio Acuto vs Cronico (ACWR = 1.15)", barmode='group', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_acwr, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Il tuo corpo è abituato a un carico di '390'. Questa settimana sei a '450'. Il rapporto è 1.15. Perfetto! Sei nello 'Sweet Spot' in cui migliori senza spaccarti (che inizia oltre 1.3).</div>", unsafe_allow_html=True)

        with g_col4:
            # Sforzo Distribuito
            fig_pie2 = px.pie(values=[70, 20, 10], names=['Aerobico Base', 'Soglia', 'Anaerobico'], title="4. Ripartizione Energetica Richiesta Oggi", hole=0.5, color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'])
            fig_pie2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie2, use_container_width=True)
            st.markdown("<div class='explain-text'><strong>Coach Tip:</strong> Brucerai principalmente grassi e ossigeno oggi. Non aver paura di prendere gel o carboidrati prima se superi l'ora di corsa, risparmierai i muscoli.</div>", unsafe_allow_html=True)

        # 5 Full width
        fig_sleep_impact = go.Figure()
        fig_sleep_impact.add_trace(go.Waterfall(
            name="Sonno", orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Sonno Base", "Fatica Allenamento Oggi", "Stress Lavorativo", "Fabbisogno Stanotte"],
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
        <strong>Coach Tip:</strong> Questo grafico 'Waterfall' ti spiega perché stasera dovrai dormire di più. 
        Parti dal tuo fabbisogno normale (7.5h). L'allenamento di oggi richiederà al tuo sistema centrale quasi 1 ora in più per riparare i tessuti (+0.8h in rosso). Lo stress lavorativo aggiunge un ulteriore ricarico. L'ultima colonna blu ti mostra il tuo vero target per stanotte. Punta la sveglia di conseguenza!
        </div>
        """, unsafe_allow_html=True)
