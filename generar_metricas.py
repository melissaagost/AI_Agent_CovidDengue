import json
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. USAMOS EXACTAMENTE LOS MISMOS PRIORS QUE EN generar_dataset_pacientes.py
_PRIOR_NINGUNA = 0.60
_PRIOR_DENGUE = 0.15
_PRIOR_COVID = 0.25

def calcular_prevalencia_estacional(es_verano: int) -> list[float]:
    """Calcula los priors ajustados según si EL PACIENTE está en verano o no."""
    p_ninguna = _PRIOR_NINGUNA
    p_dengue = _PRIOR_DENGUE
    p_covid = _PRIOR_COVID

    # Usamos la MISMA regla exacta que usamos al crear los datos
    if es_verano == 1:
        p_dengue *= 1.8
        
    # Normalizamos para que la suma dé 1.0
    total = p_ninguna + p_dengue + p_covid
    return [p_ninguna / total, p_dengue / total, p_covid / total]

def realizar_inferencia_bayesiana(evidencia: dict, es_verano: int) -> tuple[dict, str]:
    etiquetas = ['Ninguna', 'Dengue', 'COVID-19']
    
    # Pasamos la variable es_verano para calcular los priors justos
    priors = calcular_prevalencia_estacional(es_verano)

    # Las CPDs se mantienen intactas
    cpds = {
        'Fiebre': {0: [0.95, 0.10, 0.20], 1: [0.05, 0.90, 0.80]},
        'Tos': {0: [0.98, 0.80, 0.10], 1: [0.02, 0.20, 0.90]},
        'Mialgia': {0: [0.99, 0.05, 0.70], 1: [0.01, 0.95, 0.30]},
        'Perdida_Olfato': {0: [0.99, 0.98, 0.30], 1: [0.01, 0.02, 0.70]},
        'Dolor_Cabeza': {0: [0.88, 0.20, 0.50], 1: [0.12, 0.80, 0.50]},
        'Viaje_Reciente': {0: [0.95, 0.55, 0.80], 1: [0.05, 0.45, 0.20]},
        'Contacto_Positivo': {0: [0.98, 0.60, 0.65], 1: [0.02, 0.40, 0.35]},
        'Dolor_Retroocular': {0: [0.99, 0.60, 0.95], 1: [0.01, 0.40, 0.05]},
        'Sarpullido': {0: [0.98, 0.45, 0.99], 1: [0.02, 0.55, 0.01]},
        'Diarrea': {0: [0.95, 0.50, 0.70], 1: [0.05, 0.50, 0.30]}
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
    
    return dict(zip(etiquetas, probabilidades)), prediccion

def graficar_matriz(y_true, y_pred, etiquetas, titulo, nombre_archivo):
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
    print("Cargando datos y separando conjuntos de prueba...")
    df = pd.read_csv("pacientes_sinteticos.csv")
    
    # 1. DIVISIÓN CORRECTA (Train/Test Split) PARA EVITAR TRAMPAS
    X = df.drop(columns=['Enfermedad'])
    y = df['Enfermedad']
    _, X_prueba, _, y_real = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Extraemos la lista real para comparar luego
    y_real = y_real.tolist()
    etiquetas = ['Ninguna', 'Dengue', 'COVID-19']

    # 2. Predicciones con la Red Neuronal
    print("Evaluando Red Neuronal con el 20% de prueba...")
    modelo_neuronal = joblib.load("modelo_neuronal.pkl")
    y_pred_neuronal = modelo_neuronal.predict(X_prueba)

    # 3. Predicciones con la Red Bayesiana
    print("Evaluando Red Bayesiana con el mismo 20%...")
    y_pred_bayesiana = []

    for _, fila in X_prueba.iterrows():
        evidencia = {
            'Fiebre': fila['Fiebre'], 'Tos': fila['Tos'], 'Mialgia': fila['Mialgia'],
            'Perdida_Olfato': fila['Perdida_Olfato'], 'Dolor_Cabeza': fila['Dolor_Cabeza'],
            'Viaje_Reciente': fila['Viaje_Reciente'], 'Contacto_Positivo': fila['Contacto_Positivo'],
            'Dolor_Retroocular': fila['Dolor_Retroocular'], 'Sarpullido': fila['Sarpullido'],
            'Diarrea': fila['Diarrea']
        }

        # Pasamos la evidencia Y SI ESTE PACIENTE estaba en verano
        _, prediccion_final = realizar_inferencia_bayesiana(evidencia, int(fila['Verano']))
        y_pred_bayesiana.append(prediccion_final)

    # 4. Cálculo matemático
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

    print("\nGenerando gráficos de Matrices de Confusión...")
    graficar_matriz(y_real, y_pred_neuronal, etiquetas, "Matriz de Confusión - Red Neuronal", "matriz_neuronal.png")
    graficar_matriz(y_real, y_pred_bayesiana, etiquetas, "Matriz de Confusión - Red Bayesiana", "matriz_bayesiana.png")
    print("¡Proceso terminado con éxito!")

if __name__ == "__main__":
    principal()