import streamlit as st
import pandas as pd
import joblib
import json
import datetime

# Importamos el motor de la Red Bayesiana
from agente_medico import realizar_inferencia

# Configuración de la página
st.set_page_config(page_title="Sistema de Triaje Infectológico", layout="wide")

st.title("Sistema Experto de Triaje Infectológico")
st.write("Complete los datos del paciente para evaluar el riesgo de Dengue o COVID-19.")

# Cargar el modelo de la Red Neuronal
@st.cache_resource
def cargar_modelo_neuronal():
    return joblib.load("modelo_neuronal.pkl")

modelo_neuronal = cargar_modelo_neuronal()

# Cargar métricas globales precomputadas
@st.cache_resource
def cargar_metricas_modelos():
    try:
        with open("metricas_modelos.json", "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        return None

metricas_modelos = cargar_metricas_modelos()

# === CREACIÓN DE PESTAÑAS (TABS) VISUALES ===
tab1, tab2 = st.tabs(["🩺 Triaje Clínico (Paciente)", "🔬 Documentación y Métricas Globales"])

# ==========================================================
# PESTAÑA 1: LA CONSULTA DEL PACIENTE (INTERFAZ MÉDICA)
# ==========================================================
with tab1:
    st.header("Síntomas y Factores de Riesgo")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: fiebre = st.checkbox("Fiebre")
    with col_s2: tos = st.checkbox("Tos")
    with col_s3: mialgia = st.checkbox("Mialgias (Dolor muscular)")

    col_s4, col_s5, col_s6 = st.columns(3)
    with col_s4: perdida_olfato = st.checkbox("Pérdida de Olfato / Gusto")
    with col_s5: dolor_cabeza = st.checkbox("Dolor de Cabeza")
    with col_s6: viaje_reciente = st.checkbox("Viaje Reciente a zona de brote")

    col_s7, col_s8, col_s9 = st.columns(3)
    with col_s7: contacto_positivo = st.checkbox("Contacto con Persona Positiva")
    with col_s8: dolor_retroocular = st.checkbox("Dolor Retroocular (detrás de ojos)")
    with col_s9: sarpullido = st.checkbox("Sarpullido / Erupción Cutánea")

    col_s10 = st.columns(1)
    with col_s10[0]: diarrea = st.checkbox("Diarrea")

    mes_actual = datetime.datetime.now().month
    verano = 1 if mes_actual in [12, 1, 2, 3] else 0

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botón de ejecución
    if st.button("Calcular Diagnóstico", type="primary", use_container_width=True):

        st.markdown("---")
        st.markdown("<h2 style='text-align: center; color: #1f77b4;'>Resultados del Triaje</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Procesamiento en la Red Bayesiana
        evidencia_bayes = {
            'Fiebre': int(fiebre), 'Tos': int(tos), 'Mialgia': int(mialgia),
            'Perdida_Olfato': int(perdida_olfato), 'Dolor_Cabeza': int(dolor_cabeza),
            'Viaje_Reciente': int(viaje_reciente), 'Contacto_Positivo': int(contacto_positivo),
            'Dolor_Retroocular': int(dolor_retroocular), 'Sarpullido': int(sarpullido),
            'Diarrea': int(diarrea)
        }

        probs_bayes, explicacion_bayes = realizar_inferencia(evidencia_bayes)
        prediccion_bayes = max(probs_bayes, key=probs_bayes.get)
        confianza_bayes = probs_bayes[prediccion_bayes] * 100

        # Procesamiento en la Red Neuronal
        datos_paciente_nn = pd.DataFrame([{
            'Verano': verano, 'Fiebre': int(fiebre), 'Tos': int(tos), 'Mialgia': int(mialgia),
            'Perdida_Olfato': int(perdida_olfato), 'Dolor_Cabeza': int(dolor_cabeza),
            'Viaje_Reciente': int(viaje_reciente), 'Contacto_Positivo': int(contacto_positivo),
            'Dolor_Retroocular': int(dolor_retroocular), 'Sarpullido': int(sarpullido),
            'Diarrea': int(diarrea)
        }])

        prediccion_nn = modelo_neuronal.predict(datos_paciente_nn)[0]

        probs_nn = None
        confianza_nn = None
        try:
            probs_nn = modelo_neuronal.predict_proba(datos_paciente_nn)[0]
            clases_nn = modelo_neuronal.classes_
            confianza_nn = probs_nn.max() * 100
        except AttributeError:
            pass

        # Salida Visual Comparativa
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.subheader("Red Bayesiana (IA Simbólica)")
            st.metric(label="Diagnóstico Principal", value=prediccion_bayes)
            
            if prediccion_bayes == "Ninguna": st.success("Paciente sin riesgo inminente detectado.")
            else: st.error(f"⚠️ Alerta Clínica: Posible {prediccion_bayes}")

            st.write(f"**Confianza del modelo:** {confianza_bayes:.1f}%")
            st.progress(int(confianza_bayes))

            st.markdown("### Subsistema de Explicación")
            st.info(explicacion_bayes)

        with col_der:
            st.subheader("Red Neuronal (IA No Simbólica)")
            st.metric(label="Diagnóstico Principal", value=prediccion_nn)
            
            if prediccion_nn == "Ninguna": st.success("Paciente sin riesgo inminente detectado.")
            else: st.error(f"⚠️ Alerta Clínica: Posible {prediccion_nn}")

            if probs_nn is not None:
                df_probs = pd.DataFrame({"Enfermedad": clases_nn, "Probabilidad (%)": probs_nn * 100}).set_index("Enfermedad")
                st.write("**Distribución de Probabilidades Internas:**")
                st.bar_chart(df_probs)
            else:
                st.warning("Este modelo funciona con fronteras de decisión rígidas y no expone probabilidades.")

            st.markdown("### Naturaleza del Modelo")
            st.warning("Este modelo es una caja negra probabilística. Reconoce patrones complejos de forma matemática, pero carece de un mecanismo para explicar su decisión clínica.")

        # Comparativa de probabilidades para el caso actual
        st.markdown("---")
        st.subheader("Comparativa de probabilidades del caso actual")
        bayes_rows = [f"**{label}**: {probs_bayes[label] * 100:.1f}%" for label in probs_bayes]
        if probs_nn is not None:
            neur_rows = [f"**{clases_nn[i]}**: {probs_nn[i] * 100:.1f}%" for i in range(len(clases_nn))]
        else:
            neur_rows = [f"**{prediccion_nn}**: decisión sin probabilidades disponibles"]

        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.markdown("**Red Bayesiana**")
            for line in bayes_rows: st.markdown(line)
        with col_prob2:
            st.markdown("**Red Neuronal**")
            for line in neur_rows: st.markdown(line)

        if probs_nn is not None:
            if prediccion_bayes != prediccion_nn:
                st.info(f"Los motores no coinciden: Bayesiana → {prediccion_bayes}, Neuronal → {prediccion_nn}. Esto muestra diferencias en los criterios de inferencia.")
            else:
                st.info("Aunque ambos modelos eligen la misma categoría final, sus probabilidades internas difieren, indicando que ponderan los síntomas de forma distinta.")


# ==========================================================
# PESTAÑA 2: AUDITORÍA Y LABORATORIO 
# ==========================================================
with tab2:
    st.header("Resumen de Rendimiento de los Modelos")
    st.write("Métricas globales calculadas sobre el dataset sintético de 5.000 pacientes.")

    if metricas_modelos is not None:
        accuracy_bayesiana = metricas_modelos.get("accuracy_bayesiana", None)
        accuracy_neuronal = metricas_modelos.get("accuracy_neuronal", None)
        mejor_modelo = metricas_modelos.get("mejor_modelo", "N/A")

        if accuracy_bayesiana is not None and accuracy_neuronal is not None:
            diferencia = (accuracy_neuronal - accuracy_bayesiana) * 100
            diferencia_text = f"{diferencia:.1f} pts"
            if abs(diferencia) < 1e-6: diferencia_text = "Igual en este set"

            col_card1, col_card2, col_card3 = st.columns(3)
            with col_card1:
                st.metric("Red Bayesiana", f"{accuracy_bayesiana:.2f}", "Accuracy")
            with col_card2:
                st.metric("Red Neuronal", f"{accuracy_neuronal:.2f}", "Accuracy")
            with col_card3:
                st.metric("Modelo más preciso", mejor_modelo)
                st.metric("Diferencia", diferencia_text)

        st.caption("**¿Qué es el Accuracy?** Es la exactitud global del sistema. Un valor de 0.94 significa que el modelo diagnosticó correctamente al 94% de los pacientes en nuestras pruebas de laboratorio. Es el indicador principal de que la IA ha comprendido los patrones clínicos.")
    else:
        st.warning("No se pudo cargar el archivo de métricas. Ejecute generar_metricas.py")

    st.markdown("---")
    st.header("Análisis Profundo: Matrices de Confusión")
    with st.expander("Lectura de las metricas", expanded=False):
        st.markdown("""
        En la evaluación de sistemas de Inteligencia Artificial para la salud, un error no cuesta dinero, cuesta vidas. Por ello, las métricas se interpretan bajo un rigor clínico:
        
        * **Exactitud (Accuracy):** Es el porcentaje general de aciertos. Indica que de cada 100 pacientes, el modelo diagnosticó correctamente a ~94.
        * **Verdaderos Positivos (La Diagonal):** Pacientes correctamente diagnosticados con la enfermedad que realmente padecían.
        * **Falsos Negativos (Riesgo Crítico):** Ocurre cuando un paciente **tiene Dengue o COVID**, pero el sistema le diagnostica *"Ninguna"* (sano). Es el error más peligroso, ya que implica enviar a un paciente infectado a su casa sin tratamiento.
        * **Falsos Positivos (Carga Operativa):** Ocurre cuando un paciente está **Sano**, pero el sistema le diagnostica una enfermedad. No pone en riesgo la vida, pero sobrecarga al sistema de salud con análisis (hisopados/laboratorio) innecesarios.
        * **F1-Score (0.94 en este sistema):** Métrica avanzada que equilibra los falsos positivos y falsos negativos, demostrando que el modelo es altamente confiable y no adivina al azar.
        """)
    
    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.subheader("Red Bayesiana")
        try: st.image("matriz_bayesiana.png", use_container_width=True)
        except FileNotFoundError: st.warning("Falta matriz_bayesiana.png")
    with col_mat2:
        st.subheader("Red Neuronal")
        try: st.image("matriz_neuronal.png", use_container_width=True)
        except FileNotFoundError: st.warning("Falta matriz_neuronal.png")

    st.info(
        "**Conclusión de la prueba:** Ambos modelos demostraron una excelente capacidad para detectar el Dengue. "
        "Sin embargo, la Red Neuronal generó menos falsas alarmas en pacientes sanos."
    )

    st.divider()
    st.header("Laboratorio de Experimentos (Análisis Avanzado)")
    
    with st.expander("Ver Experimentos de Estacionalidad y Robustez", expanded=True):
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            st.subheader("Experimento 2: Factor Estacional")
            try: st.image("experimento_estacionalidad.png", use_container_width=True)
            except FileNotFoundError: st.warning("Falta experimento_estacionalidad.png")
            st.markdown("""
            **¿Cómo funciona el contexto epidemiológico?**
            En la IA Bayesiana, esto se conoce como **Probabilidad a Priori**. El modelo detecta el mes actual y ajusta el riesgo base antes de evaluar los síntomas. 
            Si estamos en verano (temporada alta de mosquitos en Corrientes), el modelo multiplica el riesgo base de Dengue, imitando el 'sentido común' de un médico local ante un caso dudoso.
            """)

        with col_exp2:
            st.subheader("Experimento 3: Prueba de Robustez")
            try: st.image("experimento_robustez.png", use_container_width=True)
            except FileNotFoundError: st.warning("Falta experimento_robustez.png")
            st.markdown("""
            La Red Neuronal depende de patrones completos.
            Ante la falta de información clínica, su rendimiento decae más rápido que el de la Red Bayesiana, 
            la cual logra amortiguar la incertidumbre gracias a la marginación estadística.
            """)