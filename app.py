import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime

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

# Cargar métricas globales precomputadas (accuracy) para mostrar comparativas
@st.cache_resource
def cargar_metricas_modelos():
    try:
        with open("metricas_modelos.json", "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        return None

metricas_modelos = cargar_metricas_modelos()

# Interfaz de usuario: Captura de datos
st.header("Síntomas y Factores de Riesgo")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    fiebre = st.checkbox("Fiebre")
with col_s2:
    tos = st.checkbox("Tos")
with col_s3:
    mialgia = st.checkbox("Mialgias (Dolor muscular)")

col_s4, col_s5, col_s6 = st.columns(3)
with col_s4:
    perdida_olfato = st.checkbox("Pérdida de Olfato / Gusto")
with col_s5:
    dolor_cabeza = st.checkbox("Dolor de Cabeza")
with col_s6:
    viaje_reciente = st.checkbox("Viaje Reciente a zona de brote")

col_s7, col_s8, col_s9 = st.columns(3)
with col_s7:
    contacto_positivo = st.checkbox("Contacto con Persona Positiva")
with col_s8:
    dolor_retroocular = st.checkbox("Dolor Retroocular (detrás de ojos)")
with col_s9:
    sarpullido = st.checkbox("Sarpullido / Erupción Cutánea")

col_s10 = st.columns(1)
with col_s10[0]:
    diarrea = st.checkbox("Diarrea")

mes_actual = datetime.now().month
verano = 1 if mes_actual in [12, 1, 2, 3] else 0

# Botón de ejecución
if st.button("Calcular Diagnóstico", type="primary"):

    # Procesamiento en la Red Bayesiana
    evidencia_bayes = {
        'Fiebre': int(fiebre),
        'Tos': int(tos),
        'Mialgia': int(mialgia),
        'Perdida_Olfato': int(perdida_olfato),
        'Dolor_Cabeza': int(dolor_cabeza),
        'Viaje_Reciente': int(viaje_reciente),
        'Contacto_Positivo': int(contacto_positivo),
        'Dolor_Retroocular': int(dolor_retroocular),
        'Sarpullido': int(sarpullido),
        'Diarrea': int(diarrea)
    }

    probs_bayes, explicacion_bayes = realizar_inferencia(evidencia_bayes)
    prediccion_bayes = max(probs_bayes, key=probs_bayes.get)
    confianza_bayes = probs_bayes[prediccion_bayes] * 100

    # Procesamiento en la Red Neuronal
    datos_paciente_nn = pd.DataFrame([{
        'Verano': verano,
        'Fiebre': int(fiebre),
        'Tos': int(tos),
        'Mialgia': int(mialgia),
        'Perdida_Olfato': int(perdida_olfato),
        'Dolor_Cabeza': int(dolor_cabeza),
        'Viaje_Reciente': int(viaje_reciente),
        'Contacto_Positivo': int(contacto_positivo),
        'Dolor_Retroocular': int(dolor_retroocular),
        'Sarpullido': int(sarpullido),
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
    st.header("Comparativa de Motores de Inferencia")
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("Red Bayesiana (IA Simbólica)")
        st.metric(label="Diagnóstico Principal", value=prediccion_bayes)

        # Barra de progreso visual para la confianza
        st.write(f"**Confianza del modelo:** {confianza_bayes:.1f}%")
        st.progress(int(confianza_bayes))

        st.markdown("### Subsistema de Explicación")
        st.info(explicacion_bayes)

    with col_der:
        st.subheader("Red Neuronal (IA No Simbólica)")
        st.metric(label="Diagnóstico Principal", value=prediccion_nn)

        # Gráfico de barras para las probabilidades internas
        if probs_nn is not None:
            # Formatear datos para el gráfico de Streamlit
            df_probs = pd.DataFrame({
                "Enfermedad": clases_nn,
                "Probabilidad (%)": probs_nn * 100
            }).set_index("Enfermedad")

            st.write("**Distribución de Probabilidades Internas:**")
            st.bar_chart(df_probs)
        else:
            st.warning("Este modelo funciona con fronteras de decisión rígidas y no expone probabilidades.")

        st.markdown("### Naturaleza del Modelo")
        st.warning(
            "Este modelo es una caja negra probabilística. Reconoce patrones "
            "complejos de forma matemática, pero carece de un mecanismo para explicar su decisión clínica."
        )

    # Nueva sección visual con comparativa de precisión general de los modelos
    st.markdown("---")
    st.header("Resumen de Rendimiento de los Modelos")
    st.write(
        "A continuación se muestran las métricas globales de accuracy para ambos motores de inferencia. "
        "Debajo también verás la comparativa de probabilidades del caso actual, para dejar en claro que cada motor elige diferente."
    )

    if metricas_modelos is not None:
        accuracy_bayesiana = metricas_modelos.get("accuracy_bayesiana", None)
        accuracy_neuronal = metricas_modelos.get("accuracy_neuronal", None)
        mejor_modelo = metricas_modelos.get("mejor_modelo", "N/A")

        if accuracy_bayesiana is not None and accuracy_neuronal is not None:
            diferencia = (accuracy_neuronal - accuracy_bayesiana) * 100
            diferencia_text = f"{diferencia:.1f} pts"
            if abs(diferencia) < 1e-6:
                diferencia_text = "Igual en este set"

            col_card1, col_card2, col_card3 = st.columns(3)
            with col_card1:
                st.metric("Red Bayesiana", f"{accuracy_bayesiana:.2f}", "Accuracy")
                st.info("Precisión global calculada sobre un conjunto de prueba sintético.")

            with col_card2:
                st.metric("Red Neuronal", f"{accuracy_neuronal:.2f}", "Accuracy")
                st.info("Precisión global calculada sobre un conjunto de prueba sintético.")

            with col_card3:
                st.metric("Modelo más preciso", mejor_modelo)
                st.metric("Diferencia", diferencia_text)
                if diferencia_text == "Igual en este set":
                    st.info("La accuracy global quedó empatada en la muestra sintética.")
                else:
                    st.info("Comparativa visual de los dos motores.")

            st.write("**Comentario:** La accuracy global es similar, pero esto no quiere decir que los motores calculen igual. Observa la comparativa de probabilidades actuales para ver las diferencias de cada caso.")

            # Comparativa de probabilidades para el caso actual
            st.subheader("Comparativa de probabilidades del caso actual")
            bayes_rows = [f"**{label}**: {probs_bayes[label] * 100:.1f}%" for label in probs_bayes]
            if probs_nn is not None:
                neur_rows = [f"**{clases_nn[i]}**: {probs_nn[i] * 100:.1f}%" for i in range(len(clases_nn))]
            else:
                neur_rows = [f"**{prediccion_nn}**: decisión sin probabilidades disponibles"]

            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.markdown("**Red Bayesiana**")
                for line in bayes_rows:
                    st.markdown(line)

            with col_prob2:
                st.markdown("**Red Neuronal**")
                for line in neur_rows:
                    st.markdown(line)

            if probs_nn is not None:
                if prediccion_bayes != prediccion_nn:
                    st.info(
                        f"Los dos motores no coinciden en el diagnóstico final: Bayesiana → {prediccion_bayes}, Neuronal → {prediccion_nn}. "
                        "Esto muestra claramente que las decisiones son diferentes aun cuando la accuracy global es igual."
                    )
                else:
                    st.info(
                        "Aunque ambos modelos eligen la misma categoría final, sus probabilidades internas difieren, lo cual indica que usan criterios distintos."
                    )
        else:
            st.warning("No se encontraron métricas completas para mostrar la comparativa de modelos.")
    else:
        st.warning("No se pudo cargar el archivo de métricas. Ejecute generar_metricas.py para generar metricas_modelos.json.")

    # Auditoría y Trazabilidad (Gobierno de Datos)
    st.divider()
    with st.expander("Auditoría de Datos (Trazabilidad de Entrada)"):
        st.write("Estructuras de datos inyectadas a los motores de inferencia. Útil para validar la exactitud del procesamiento.")
        col_audit1, col_audit2 = st.columns(2)

        with col_audit1:
            st.write("**Entrada Red Bayesiana (Diccionario de Evidencia):**")
            st.json(evidencia_bayes)

        with col_audit2:
            st.write("**Entrada Red Neuronal (DataFrame tabular):**")
            st.dataframe(datos_paciente_nn)