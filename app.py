import streamlit as st
from agente_medico import realizar_inferencia

st.set_page_config(page_title="Sistema Experto Médico", layout="wide")

st.title("Clasificador de Riesgo: Dengue o COVID-19")
st.write("Ingrese los datos y síntomas del paciente para el análisis automatizado.")

# Percepción: Entrada de datos (Sensores del agente)
with st.sidebar:
    st.header("Percepción: Datos del Paciente")
    
    st.subheader("Síntomas Clínicos")
    f = st.checkbox("Fiebre")
    t = st.checkbox("Tos")
    m = st.checkbox("Dolores Musculares (Mialgia)")
    
    st.subheader("Antecedentes Epidemiológicos")
    v = st.checkbox("Viaje reciente")
    c = st.checkbox("Contacto con paciente de Dengue")

if st.button("Diagnosticar"):
    # Mapeo de percepciones a evidencia binaria para el motor de inferencia
    evidencia = {
        'Fiebre': 1 if f else 0,
        'Tos': 1 if t else 0,
        'Mialgia': 1 if m else 0,
        'Viaje_Reciente': 1 if v else 0,
        'Contacto_Dengue': 1 if c else 0
    }
    
    probs, explicacion = realizar_inferencia(evidencia)
    
    # Acción: Mostrar resultados y recomendaciones (Actuadores del agente)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Probabilidades (Modelo Bayesiano)")
        st.bar_chart(probs)
        
    with col2:
        st.subheader("Subsistema de Explicación")
        st.info(explicacion)
        
        # Recomendación basada en el flujo de Acción del documento
        st.subheader("Acciones Recomendadas")
        if probs["Dengue"] > max(probs["COVID-19"], probs["Ninguna"]):
            st.warning("Acción: Derivar a sala de aislamiento y proporcionar información básica sobre Dengue y control de vectores.")
        elif probs["COVID-19"] > max(probs["Dengue"], probs["Ninguna"]):
            st.warning("Acción: Derivar a sala de aislamiento y proporcionar información básica sobre protocolos respiratorios preventivos.")
        else:
            st.success("Acción: Atención general en sala de espera estándar.")