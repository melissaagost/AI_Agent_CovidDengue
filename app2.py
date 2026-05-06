import streamlit as st
import plotly.express as px
import pandas as pd
from agente_medico import realizar_inferencia
from datetime import datetime

# Configuración de página con estilo médico
st.set_page_config(page_title="Sistema Experto de Triaje Infectológico", layout="wide")

# CSS personalizado para elegancia y sobriedad
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #004a7c; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("Clasificador de Riesgo: Dengue o COVID-19")
st.write(f"Protocolo de Emergencias - Corrientes, {datetime.now().strftime('%d/%m/%Y')}")
st.markdown("---")

# --- SENSORES: Entrada de datos en Sidebar ---
# Percepción: Entrada de datos (Sensores del agente)
with st.sidebar:
    st.header("Percepción: Datos del Paciente")
    
    st.subheader(" Datos de Contexto")
    zona_corrientes = st.checkbox("¿Reside en zona de brote activo?", value=True)
    viaje = st.checkbox("¿Viaje reciente a zonas endémicas?")
    contacto = st.checkbox("¿Contacto estrecho con Dengue positivo?")

    st.subheader(" Síntomas Sistémicos")
    f = st.checkbox("Fiebre (Alta/Abrupta)")
    m = st.checkbox("Mialgia (Dolores Musculares)")
    
    st.subheader(" Síntomas Respiratorios")
    t = st.checkbox("Tos Seca")
    g = st.checkbox("Dolor de Garganta / Congestión")
    o = st.checkbox("Anosmia (Pérdida de Olfato/Gusto)")

    st.subheader(" Síntomas Específicos")
    r = st.checkbox("Dolor Retroocular")
    s = st.checkbox("Sarpullido / Exantema")

# --- PROCESAMIENTO: Motor de Inferencia ---
# Mapeo de percepciones a evidencia binaria para el motor de inferencia
evidencia = {
    'Fiebre': 1 if f else 0, 'Tos': 1 if t else 0, 'Mialgia': 1 if m else 0,
    'Viaje_Reciente': 1 if viaje else 0, 'Contacto_Dengue': 1 if contacto else 0,
    'Dolor_Retroocular': 1 if r else 0, 'Perdida_Olfato': 1 if o else 0,
    'Sarpullido': 1 if s else 0, 'Garganta_Congestion': 1 if g else 0
}

# --- ACTUADORES: Visualización y Acciones ---
if any(evidencia.values()):
    probs, explicacion = realizar_inferencia(evidencia)
    
    # Determinamos el ganador
    ganador = max(probs, key=probs.get)
    color_ganador = "#e74c3c" if ganador != "Ninguna" else "#27ae60"

    # Fila de Métricas Principales
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Diagnóstico Sugerido", ganador)
    with col_m2:
        st.metric("Confianza del Motor", f"{probs[ganador]*100:.1f}%")
    with col_m3:
        st.metric("Urgencia", "ALTA" if ganador != "Ninguna" else "NORMAL")

    st.markdown("###  Análisis Probabilístico Comparativo")
    
    # Gráfico Avanzado con Plotly
    df_plot = pd.DataFrame({
        'Patología': list(probs.keys()),
        'Probabilidad': [v * 100 for v in probs.values()]
    })
    
    fig = px.bar(df_plot, x='Probabilidad', y='Patología', orientation='h',
                 color='Probabilidad', color_continuous_scale='teal',
                 text_auto='.1f', title="Distribución de Probabilidad Bayesiana")
    fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.subheader(" Subsistema de Explicación")
        st.success(explicacion)

    with col_res2:
        st.subheader("Acciones Recomendadas")
        if ganador == "Dengue":
            st.error("**DERIVACIÓN:** Infectología - Control de Plaquetas.\n\n**INSTRUCCIÓN:** Hidratación oral abundante. NO administrar Aspirina o Ibuprofeno. Derivar a sala de aislamiento y proporcionar información básica sobre Dengue y control de vectores.")
        elif ganador == "COVID-19":
            st.error("**DERIVACIÓN:** Aislamiento Respiratorio - Test PCR.\n\n**INSTRUCCIÓN:** Uso obligatorio de barbijo N95 y monitoreo de saturación de oxígeno. Derivar a sala de aislamiento y proporcionar información básica sobre protocolos respiratorios preventivos.")
        else:
            st.info("**ATENCIÓN GENERAL:** El paciente no presenta indicadores críticos de brote. Proceder a clínica médica estándar.")

else:
    st.info(" Bienvenida/o al Sistema Experto. Por favor, cargue los síntomas en el panel lateral para iniciar el diagnóstico.")