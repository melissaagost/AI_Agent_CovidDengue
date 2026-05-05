import streamlit as st
from agente_medico import realizar_inferencia

st.set_page_config(page_title="Sistema Experto Médico", layout="wide")

st.title("¿Dengue o Covid-19?")
st.write("Seleccione sus síntomas")

# Percepción: Entrada de datos
with st.sidebar:
    st.header("Datos del Paciente")
    f = st.checkbox("Fiebre")
    t = st.checkbox("Tos")
    m = st.checkbox("Dolores Musculares (Mialgia)")

if st.button("Diagnosticar"):
    # Mapeo de percepciones a evidencia binaria
    evidencia = {
        'Fiebre': 1 if f else 0,
        'Tos': 1 if t else 0,
        'Mialgia': 1 if m else 0
    }
    
    probs, explicacion = realizar_inferencia(evidencia)
    
    # Acción: Mostrar resultados y recomendaciones
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resultados Estadísticos")
        st.bar_chart(probs)
        #se puede agregar otra info visual
        
    with col2:
        st.subheader("Explicación del Agente")
        st.info(explicacion)
        
        # Recomendación basada en reglas adicionales
        if probs["Dengue"] > 0.4 or probs["COVID-19"] > 0.4:
            st.warning("Acción Recomendada: Derivar a sala de aislamiento.")
            #agregar otras acciones recomendadas
        else:
            st.success("Acción Recomendada: Atención general.")