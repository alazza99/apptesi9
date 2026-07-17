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

# 1. Configurazione Pagina (Deve essere il primo comando)
st.set_page_config(page_title="RunAI Coach", page_icon="⭕", layout="wide", initial_sidebar_state="expanded")

# 2. CSS Custom: Stile WHOOP (Dark Mode, Card arrotondate, Colori Neon)
st.markdown("""
<style>
    /* Sfondo e Testo Generale */
    .stApp { background-color: #000000; color: #FFFFFF; font-family: 'Inter', 'Segoe UI', sans-serif; }
    
    /* Nascondi header di default di Streamlit */
    header { visibility: hidden; }
    
    /* Stile Sidebar */
    [data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #2C2C2E; }
    
    /* Titoli */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    
    /* Card personalizzate stile Whoop */
    .whoop-card {
        background-color: #1C1C1E;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #2C2C2E;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .whoop-title { font-size: 0.9rem; color: #8E8E93; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; font-weight: 600;}
    .whoop-value { font-size: 2.5rem; font-weight: 800; margin-bottom: 5px; }
    
    /* Colori Semaforo */
    .text-green { color: #32D74B !important; }
    .text-yellow { color: #FFD60A !important; }
    .text-red { color: #FF453A !important; }
    .text-blue { color: #0A84FF !important; }
    .text-cyan { color: #64D2FF !important; }
    
    /* Rimuovi sfondi input Streamlit */
    .stTextInput>div>div>input { background-color: #2C2C2E; color: white; border: none; border-radius: 10px;}
    .stSelectbox>div>div>div { background-color: #2C2C2E; color: white; border: none; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Imposta il tema scuro di default per Plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"

# 3. Generazione Dati
@st.cache_data
def genera_dati():
    np.random.seed(42)
    n = 90
    
    velocita = np.random.uniform(9, 16, n)
    distanza = np.random.uniform(5, 25, n)
    ore_sonno = np.random.uniform(4.5, 9, n)
    stress_lavoro = np.random.randint(1, 11, n)
    
    fc_media = 100 + (velocita * 3) + (distanza * 0.5) + np.random.normal(0, 5, n)
    fc_media = np.clip(fc_media, 80, 200)
    
    # Adattamento in stile WHOOP: Sforzo (Strain) su scala 0-21
    strain_base = (distanza * 0.5) + (velocita * 0.3) + np.random.normal(0, 1, n)
    strain = np.clip(np.round(strain_base, 1), 4.0, 20.9)
    
    # Recupero (Recovery) su scala 0-100%
    recovery_base = 100 - (stress_lavoro * 3) - ((8 - ore_sonno) * 10) + np.random.normal(0, 5, n)
    recovery = np.clip(np.round(recovery_base), 1, 99)
    
    df = pd.DataFrame({
        'Giorno': pd.date_range(end=pd.Timestamp.today(), periods=n),
        'Distanza (km)': np.round(distanza, 1),
        'Velocità (km/h)': np.round(velocita, 1),
        'FC Media': np.round(fc_media),
        'Strain (0-21)': strain,
        'Recovery (%)': recovery,
        'Ore Sonno': np.round(ore_sonno, 1),
        'Fabbisogno Sonno': np.round(np.random.uniform(7.5, 8.5, n), 1),
        'Stress Lavoro': stress_lavoro
    })
    
    df['Sleep Performance (%)'] = np.clip(np.round((df['Ore Sonno'] / df['Fabbisogno Sonno']) * 100), 0, 100)
    # Rischio infortunio basato sulle metriche Whoop
    df['Rischio Infortunio'] = np.where((df['Strain (0-21)'] > 16) & (df['Recovery (%)'] < 33), 1, 0)
    
    return df

# Inizializzazione Session State
if 'dati' not in st.session_state:
    st.session_state.dati = genera_dati()
    st.session_state.journal_compilato = False

# 4. Componente UI: Gauge Circolare stile Whoop
def create_whoop_gauge(value, max_value, title, color, suffix="", is_strain=False):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'font': {'size': 50, 'color': 'white', 'family': 'Inter'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_value], 'visible': False},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#2C2C2E",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=title, x=0.5, y=0.1, showarrow=False, font=dict(size=16, color="#8E8E93", family="Inter"))]
    )
    return fig

# Helper per i colori del Recovery
def get_recovery_color(val):
    if val >= 67: return "#32D74B" # Verde
    if val >= 34: return "#FFD60A" # Giallo
    return "#FF453A" # Rosso

# 5. Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>⭕ RunAI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color:#8E8E93; font-size:0.9em;'>COACHING ENGINE</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    pagina = st.radio("MENU", 
        ["⭕ Overview Oggi", "📉 Trend & Statistiche", "🤖 Coach & ML"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<div class='whoop-title'>DEVICE STATUS</div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color:#1C1C1E; padding:15px; border-radius:15px; border: 1px solid #2C2C2E;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                <span style='color:#8E8E93;'>Battery</span>
                <span style='color:#32D74B; font-weight:bold;'>87% 🔋</span>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='color:#8E8E93;'>Live HR</span>
                <span style='color:white; font-weight:bold;'>58 bpm ❤️</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- PAGINA 1: OVERVIEW OGGI ---
if pagina == "⭕ Overview Oggi":
    st.markdown("<h2>Today's Overview</h2>", unsafe_allow_html=True)
    
    if not st.session_state.journal_compilato:
        st.markdown("""
            <div class='whoop-card'>
                <h3 style='margin-top:0;'>📝 WHOOP Journal</h3>
                <p style='color:#8E8E93;'>Compila i dati della giornata per calcolare il tuo stato in tempo reale.</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("journal"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ore_sonno = st.number_input("Ore di sonno stanotte", 2.0, 12.0, 6.5, 0.5)
            with col2:
                stress = st.slider("Stress Lavoro (1-10)", 1, 10, 6)
            with col3:
                rpe_allenamento = st.slider("Intensità Allenamento (RPE 0-10)", 0, 10, 7)
            
            submit = st.form_submit_button("CALCOLA METRICHE", use_container_width=True)
            
            if submit:
                # Calcoli stile Whoop
                rec = max(1, min(99, int(100 - (stress * 4) - ((8 - ore_sonno) * 12))))
                # Mappiamo RPE su Scala Strain (0-21) Logaritmica approssimata
                strain = round(min(20.9, rpe_allenamento * 1.8 + (10 - rec/10)*0.5), 1)
                sleep_perf = min(100, int((ore_sonno / 8.0) * 100))
                
                st.session_state.today_metrics = {'recovery': rec, 'strain': strain, 'sleep': sleep_perf}
                st.session_state.journal_compilato = True
                st.rerun()

    if st.session_state.journal_compilato:
        m = st.session_state.today_metrics
        rec_color = get_recovery_color(m['recovery'])
        
        # Le 3 Metriche Circolari
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(create_whoop_gauge(m['recovery'], 100, "RECOVERY", rec_color, "%"), use_container_width=True)
        with col2:
            st.plotly_chart(create_whoop_gauge(m['strain'], 21, "DAY STRAIN", "#0A84FF", "", is_strain=True), use_container_width=True)
        with col3:
            st.plotly_chart(create_whoop_gauge(m['sleep'], 100, "SLEEP PERFORMANCE", "#64D2FF", "%"), use_container_width=True)

        st.markdown("---")
        
        # Actionable Insights stile Whoop
        st.markdown("<h3>Coach Insights</h3>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            if m['recovery'] >= 67:
                msg, color_class = "Corpo primato per sforzi elevati. Ottimo giorno per spingere o per una gara.", "text-green"
            elif m['recovery'] >= 34:
                msg, color_class = "Il corpo sta lavorando per recuperare. Mantieni lo sforzo (Strain) sotto 14.0 oggi.", "text-yellow"
            else:
                msg, color_class = "Riposo attivo necessario. Priorizza il sonno e idratazione. Mantieni lo Strain sotto 10.0.", "text-red"
                
            st.markdown(f"""
                <div class='whoop-card'>
                    <div class='whoop-title'>RECOVERY INSIGHT</div>
                    <p style='font-size: 1.1rem; line-height:1.5;'><span class='{color_class}'>●</span> {msg}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown(f"""
                <div class='whoop-card'>
                    <div class='whoop-title'>SLEEP COACH</div>
                    <p style='font-size: 1.1rem; line-height:1.5;'>Hai dormito il <span class='text-cyan'>{m['sleep']}%</span> del tuo fabbisogno. Per massimizzare il recupero domani, il tuo target di sonno stanotte è di <b>8.5 ore</b>.</p>
                </div>
            """, unsafe_allow_html=True)

# --- PAGINA 2: TREND E STATISTICHE ---
elif pagina == "📉 Trend & Statistiche":
    st.markdown("<h2>Trends (Last 90 Days)</h2>", unsafe_allow_html=True)
    df = st.session_state.dati
    
    # Grafico Recovery vs Strain (Classico WHOOP)
    st.markdown("""
        <div class='whoop-card'>
            <div class='whoop-title'>STRAIN VS RECOVERY</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Definiamo i colori per il grafico a barre del recovery
    colors = [get_recovery_color(val) for val in df['Recovery (%)'].tail(30)]
    
    fig_combo = go.Figure()
    # Barre Recovery
    fig_combo.add_trace(go.Bar(
        x=df['Giorno'].tail(30), y=df['Recovery (%)'].tail(30),
        name='Recovery (%)', marker_color=colors, opacity=0.8,
        yaxis='y'
    ))
    # Linea Strain
    fig_combo.add_trace(go.Scatter(
        x=df['Giorno'].tail(30), y=df['Strain (0-21)'].tail(30),
        name='Day Strain', mode='lines+markers',
        line=dict(color='#0A84FF', width=3),
        marker=dict(size=8, color='white', line=dict(color='#0A84FF', width=2)),
        yaxis='y2'
    ))
    
    fig_combo.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis=dict(title="Recovery %", range=[0, 100], gridcolor='#2C2C2E'),
        yaxis2=dict(title="Strain", range=[0, 21], overlaying='y', side='right', showgrid=False)
    )
    st.plotly_chart(fig_combo, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='whoop-card'>", unsafe_allow_html=True)
        st.markdown("<div class='whoop-title'>SLEEP CONSISTENCY</div>", unsafe_allow_html=True)
        fig_sleep = px.line(df.tail(30), x='Giorno', y=['Ore Sonno', 'Fabbisogno Sonno'], 
                           color_discrete_sequence=['#64D2FF', '#8E8E93'])
        fig_sleep.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               yaxis_title="Ore", legend_title=None, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig_sleep, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='whoop-card'>", unsafe_allow_html=True)
        st.markdown("<div class='whoop-title'>HEART RATE ZONES</div>", unsafe_allow_html=True)
        fig_hr = px.scatter(df.tail(30), x='Strain (0-21)', y='FC Media', color='Recovery (%)',
                           color_continuous_scale=['#FF453A', '#FFD60A', '#32D74B'])
        fig_hr.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGINA 3: MACHINE LEARNING ---
elif pagina == "🤖 Coach & ML":
    st.markdown("<h2>AI Health Monitor</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='whoop-card' style='border-left: 4px solid #0A84FF;'>
            <h3 style='margin-top:0;'>Predizione Rischio Infortuni</h3>
            <p style='color:#8E8E93;'>Il nostro modello RandomForest analizza costantemente i tuoi dati storici di Strain, Sonno e Recovery per prevedere i giorni in cui il tuo corpo è statisticamente più vulnerabile.</p>
        </div>
    """, unsafe_allow_html=True)
    
    df = st.session_state.dati.copy()
    X = df[['Strain (0-21)', 'Sleep Performance (%)', 'Stress Lavoro', 'FC Media']].values
    y = df['Rischio Infortunio'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_scaled, y)
    
    importances = rf.feature_importances_
    features = ['Strain Giornaliero', 'Qualità Sonno (%)', 'Stress Mentale', 'Frequenza Cardiaca']
    
    col_ml1, col_ml2 = st.columns(2)
    
    with col_ml1:
        st.markdown("<div class='whoop-title'>IMPATTO DEI PARAMETRI (FEATURE IMPORTANCE)</div>", unsafe_allow_html=True)
        fig_imp = go.Figure(go.Bar(
            x=[x*100 for x in importances],
            y=features,
            orientation='h',
            marker=dict(color='#0A84FF'),
            text=[f'{x*100:.1f}%' for x in importances], textposition='auto'
        ))
        fig_imp.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis_title="Impatto (%)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_ml2:
        st.markdown("<div class='whoop-title'>STATO ATTUALE DEL MODELLO</div>", unsafe_allow_html=True)
        
        if st.session_state.journal_compilato:
            m = st.session_state.today_metrics
            # Array: Strain, Sleep Perf, Stress (stimato a 6), FC (stimata a 140)
            today_data = np.array([[m['strain'], m['sleep'], 6, 140]])
            today_scaled = scaler.transform(today_data)
            prob_infortunio = rf.predict_proba(today_scaled)[0][1] * 100
            
            color = "#FF453A" if prob_infortunio > 50 else "#FFD60A" if prob_infortunio > 20 else "#32D74B"
            
            st.markdown(f"""
                <div class='whoop-card' style='text-align:center; padding: 40px 20px;'>
                    <p style='color:#8E8E93; font-size:1.2rem; margin:0;'>PROBABILITÀ INFORTUNIO OGGI</p>
                    <p style='font-size:4rem; font-weight:800; color:{color}; margin:10px 0;'>{prob_infortunio:.1f}%</p>
                    <p style='color:white; font-size:1rem;'>Basato sulle tue metriche di Recovery e Strain odierne.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Compila il WHOOP Journal nella tab 'Overview Oggi' per vedere la predizione in tempo reale.")
