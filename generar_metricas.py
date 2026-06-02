#Tomar un subconjunto de los pacientes ficticios y hacer que tanto la Red Bayesiana como la Red Neuronal intenten diagnosticarlos.
# Calcular matemáticamente cuántas veces acertó cada modelo y generar las imágenes de las Matrices de Confusión (gráficos que muestran los falsos positivos y falsos negativos). Estas imágenes irán directo a tu informe IEEE.

import json
import math
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg') # Fuerza el renderizado en segundo plano sin usar Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Copia de los mismos priors y CPDs definidos en agente_medico.py
# para poder calcular la inferencia bayesiana aquí sin importar pgmpy.
_PRIOR_NINGUNA = 0.65
_PRIOR_DENGUE = 0.15
_PRIOR_COVID = 0.20

def _cargar_prior_covid() -> float:
    path = "dataset_covid_global.csv"
    df = pd.read_csv(path)
    letalidad = df['Deaths'].sum() / df['Confirmed'].sum()
    prior = 0.10 + (letalidad / 0.10) * 0.25
    return float(min(max(prior, 0.10), 0.35))


def _cargar_prior_dengue_corrientes() -> float:
    path = "dataset_dengue_y_zika_2025_2026.csv"
    df = pd.read_csv(path, sep=';', encoding='latin-1')
    df_dengue = df[df['evento'] == 'Dengue']
    total_nacional = df_dengue['cantidad'].sum()
    casos_corrientes = df_dengue[
        df_dengue['provincia_residencia'].str.lower() == 'corrientes'
    ]['cantidad'].sum()

    if total_nacional == 0:
        return _PRIOR_DENGUE

    proporcion = casos_corrientes / total_nacional
    prior = 0.15 + proporcion * 10
    return float(min(max(prior, 0.05), 0.40))


def calcular_prevalencia_estacional() -> list[float]:
    try:
        p_covid = _cargar_prior_covid()
        p_dengue = _cargar_prior_dengue_corrientes()
        p_ninguna = max(1.0 - p_covid - p_dengue, 0.05)
    except Exception:
        p_covid = _PRIOR_COVID
        p_dengue = _PRIOR_DENGUE
        p_ninguna = _PRIOR_NINGUNA

    mes_actual = pd.Timestamp.now().month
    if mes_actual in [12, 1, 2, 3]:
        p_dengue *= 1.8
        p_ninguna *= 0.85

    total = p_ninguna + p_dengue + p_covid
    return [p_ninguna / total, p_dengue / total, p_covid / total]


def realizar_inferencia_bayesiana(evidencia: dict) -> tuple[dict, str]:
    etiquetas = ['Ninguna', 'Dengue', 'COVID-19']
    priors = calcular_prevalencia_estacional()

    cpds = {
        'Fiebre': {
            0: [0.95, 0.10, 0.20],
            1: [0.05, 0.90, 0.80]
        },
        'Tos': {
            0: [0.98, 0.80, 0.10],
            1: [0.02, 0.20, 0.90]
        },
        'Mialgia': {
            0: [0.99, 0.05, 0.70],
            1: [0.01, 0.95, 0.30]
        },
        'Perdida_Olfato': {
            0: [0.99, 0.98, 0.30],
            1: [0.01, 0.02, 0.70]
        },
        'Dolor_Cabeza': {
            0: [0.88, 0.20, 0.50],
            1: [0.12, 0.80, 0.50]
        },
        'Viaje_Reciente': {
            0: [0.95, 0.55, 0.80],
            1: [0.05, 0.45, 0.20]
        },
        'Contacto_Positivo': {
            0: [0.98, 0.60, 0.65],
            1: [0.02, 0.40, 0.35]
        },
        'Dolor_Retroocular': {
            0: [0.99, 0.60, 0.95],
            1: [0.01, 0.40, 0.05]
        },
        'Sarpullido': {
            0: [0.98, 0.45, 0.99],
            1: [0.02, 0.55, 0.01]
        },
        'Diarrea': {
            0: [0.95, 0.50, 0.70],
            1: [0.05, 0.50, 0.30]
        }
    }

    scores = []
    for i in range(3):
        score = priors[i]
        for feature, value in evidencia.items():
            if feature not in cpds:
                continue
            score *= cpds[feature][value][i]
        scores.append(score)

    total = sum(scores)
    probabilidades = [s / total if total > 0 else 0.0 for s in scores]
    prediccion = etiquetas[int(max(range(3), key=lambda j: probabilidades[j]))]
    confianza = max(probabilidades) * 100

    explicacion = (
        f"Diagnóstico sugerido: **{prediccion}** con una probabilidad del **{confianza:.1f}%**."
    )
    return dict(zip(etiquetas, probabilidades)), explicacion

def graficar_matriz(y_true, y_pred, etiquetas, titulo, nombre_archivo):
    """Genera y guarda la imagen de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title(titulo)
    plt.ylabel('Diagnóstico Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    plt.close()

def principal():
    print("Cargando datos y modelos...")
    # 1. Cargar el dataset y tomar un subconjunto (ej. 200 pacientes) para no demorar la prueba
    df = pd.read_csv("pacientes_sinteticos.csv")
    df_prueba = df.sample(n=200, random_state=42)

    X_prueba = df_prueba.drop(columns=['Enfermedad'])
    y_real = df_prueba['Enfermedad'].tolist()
    etiquetas = ['Ninguna', 'Dengue', 'COVID-19']

    # 2. Predicciones con la Red Neuronal
    print("Evaluando Red Neuronal...")
    modelo_neuronal = joblib.load("modelo_neuronal.pkl")
    y_pred_neuronal = modelo_neuronal.predict(X_prueba)

    # 3. Predicciones con la Red Bayesiana
    print("Evaluando Red Bayesiana...")
    y_pred_bayesiana = []

    for _, fila in X_prueba.iterrows():
        # Diccionario de evidencia ajustado a los 10 síntomas
        evidencia = {
            'Fiebre': fila['Fiebre'],
            'Tos': fila['Tos'],
            'Mialgia': fila['Mialgia'],
            'Perdida_Olfato': fila['Perdida_Olfato'],
            'Dolor_Cabeza': fila['Dolor_Cabeza'],
            'Viaje_Reciente': fila['Viaje_Reciente'],
            'Contacto_Positivo': fila['Contacto_Positivo'],
            'Dolor_Retroocular': fila['Dolor_Retroocular'],
            'Sarpullido': fila['Sarpullido'],
            'Diarrea': fila['Diarrea']
        }

        # 1. Recibir el diccionario de probabilidades usando la inferencia bayesiana local
        diccionario_probs, _ = realizar_inferencia_bayesiana(evidencia)

        # 2. Extraer la llave (nombre de enfermedad) con el valor más alto
        prediccion_final = max(diccionario_probs, key=diccionario_probs.get)

        # 3. Guardar el string resultante en la lista
        y_pred_bayesiana.append(prediccion_final)

    # 4. Cálculo matemático y reportes (Consola)
    accuracy_neuronal = accuracy_score(y_real, y_pred_neuronal)
    accuracy_bayesiana = accuracy_score(y_real, y_pred_bayesiana)

    print("\n" + "="*50)
    print("RESULTADOS: RED NEURONAL (IA No Simbólica)")
    print("="*50)
    print(f"Precisión General (Accuracy): {accuracy_neuronal:.2f}")
    print(classification_report(y_real, y_pred_neuronal, target_names=etiquetas))

    print("\n" + "="*50)
    print("RESULTADOS: RED BAYESIANA (IA Simbólica)")
    print("="*50)
    print(f"Precisión General (Accuracy): {accuracy_bayesiana:.2f}")
    print(classification_report(y_real, y_pred_bayesiana, target_names=etiquetas))

    # Guardar métricas globales para uso en la interfaz Streamlit
    metricas = {
        "accuracy_neuronal": accuracy_neuronal,
        "accuracy_bayesiana": accuracy_bayesiana,
        "mejor_modelo": "Red Neuronal" if accuracy_neuronal >= accuracy_bayesiana else "Red Bayesiana"
    }

    with open("metricas_modelos.json", "w", encoding="utf-8") as archivo_metricas:
        json.dump(metricas, archivo_metricas, indent=2)

    print("\nSe generó el archivo 'metricas_modelos.json' con las métricas de precisión de ambos modelos.")

    # 5. Generar las imágenes para el informe IEEE
    print("\nGenerando gráficos de Matrices de Confusión...")
    graficar_matriz(y_real, y_pred_neuronal, etiquetas,
                    "Matriz de Confusión - Red Neuronal",
                    "matriz_neuronal.png")

    graficar_matriz(y_real, y_pred_bayesiana, etiquetas,
                    "Matriz de Confusión - Red Bayesiana",
                    "matriz_bayesiana.png")

    print("¡Proceso terminado! Las imágenes .png se han guardado en tu directorio.")

if __name__ == "__main__":
    principal()