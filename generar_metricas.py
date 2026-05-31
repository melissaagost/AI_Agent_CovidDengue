#Tomar un subconjunto de los pacientes ficticios y hacer que tanto la Red Bayesiana como la Red Neuronal intenten diagnosticarlos.
# Calcular matemáticamente cuántas veces acertó cada modelo y generar las imágenes de las Matrices de Confusión (gráficos que muestran los falsos positivos y falsos negativos). Estas imágenes irán directo a tu informe IEEE.

import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg') # Fuerza el renderizado en segundo plano sin usar Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Importamos la función de tu Red Bayesiana
# Asegúrate de que el nombre de la función y las salidas coincidan con tu agente_medico.py
from agente_medico import realizar_inferencia

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
        # Diccionario de evidencia ajustado a tu CSV actual
        evidencia = {
            'Fiebre': fila['Fiebre'],
            'Tos': fila['Tos'],
            'Mialgia': fila['Mialgia'],
            'Perdida_Olfato': fila['Perdida_Olfato'],
            'Viaje_Reciente': fila['Viaje_Reciente']
        }

        # 1. Recibir el diccionario de probabilidades
        diccionario_probs, _ = realizar_inferencia(evidencia)

        # 2. Extraer la llave (nombre de enfermedad) con el valor más alto
        prediccion_final = max(diccionario_probs, key=diccionario_probs.get)

        # 3. Guardar el string resultante en la lista
        y_pred_bayesiana.append(prediccion_final)

    # 4. Cálculo matemático y reportes (Consola)
    print("\n" + "="*50)
    print("RESULTADOS: RED NEURONAL (IA No Simbólica)")
    print("="*50)
    print(f"Precisión General (Accuracy): {accuracy_score(y_real, y_pred_neuronal):.2f}")
    print(classification_report(y_real, y_pred_neuronal, target_names=etiquetas))

    print("\n" + "="*50)
    print("RESULTADOS: RED BAYESIANA (IA Simbólica)")
    print("="*50)
    print(f"Precisión General (Accuracy): {accuracy_score(y_real, y_pred_bayesiana):.2f}")
    print(classification_report(y_real, y_pred_bayesiana, target_names=etiquetas))

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