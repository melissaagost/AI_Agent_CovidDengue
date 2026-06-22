import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Para evitar errores de interfaz gráfica
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from unittest.mock import patch
from agente_medico import realizar_inferencia

# ==========================================
# EXPERIMENTO 2: IMPACTO DE LA ESTACIONALIDAD
# ==========================================
def experimento_estacionalidad():
    import importlib
    import agente_medico
    from unittest.mock import patch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n--- Ejecutando Experimento 2: Estacionalidad ---")
    
    # Paciente de prueba (Síntomas cruzados Dengue/COVID para generar DUDA)
    # Al estar empatado clínicamente, la estación del año será el desempate.
    evidencia_base = {
        'Fiebre': 1, 
        'Tos': 1,             # Sube los puntos de COVID
        'Mialgia': 1,         # Sube los puntos de Dengue
        'Perdida_Olfato': 0, 
        'Dolor_Cabeza': 1,
        'Viaje_Reciente': 1,  # Factor de riesgo ambiguo
        'Contacto_Positivo': 0,
        'Dolor_Retroocular': 0, 
        'Sarpullido': 0,
        'Diarrea': 0
    }

    resultados_estacion = {'Verano': [], 'Invierno': []}

    # Test en Verano (Meses: 12, 1, 2, 3)
    for mes in [12, 1, 2, 3]:
        with patch('agente_medico.datetime.datetime') as mock_dt:
            mock_dt.now.return_value.month = mes
            # ¡LA CLAVE! Obliga a reconstruir la red con el mes falso
            importlib.reload(agente_medico) 
            probs, _ = agente_medico.realizar_inferencia(evidencia_base)
            resultados_estacion['Verano'].append(probs['Dengue'])

    # Test en Invierno (Meses: 6, 7, 8, 9)
    for mes in [6, 7, 8, 9]:
        with patch('agente_medico.datetime') as mock_dt:
            mock_dt.now.return_value.month = mes
            # ¡LA CLAVE! Obliga a reconstruir la red con el mes falso
            importlib.reload(agente_medico) 
            probs, _ = agente_medico.realizar_inferencia(evidencia_base)
            resultados_estacion['Invierno'].append(probs['Dengue'])

    # Cálculos
    promedio_verano = sum(resultados_estacion['Verano']) / len(resultados_estacion['Verano'])
    promedio_invierno = sum(resultados_estacion['Invierno']) / len(resultados_estacion['Invierno'])

    print(f"Probabilidad de Dengue en Verano: {promedio_verano:.3f} ({promedio_verano*100:.1f}%)")
    print(f"Probabilidad de Dengue en Invierno: {promedio_invierno:.3f} ({promedio_invierno*100:.1f}%)")
    print(f"Incremento de riesgo estacional: {(promedio_verano/promedio_invierno):.1f}x veces mayor")

    # Gráfico
    plt.figure(figsize=(7, 5))
    barras = plt.bar(['Verano', 'Invierno'], [promedio_verano, promedio_invierno], color=['#e74c3c', '#3498db'])
    plt.title('Impacto Estacional en el Diagnóstico de Dengue (Red Bayesiana)')
    plt.ylabel('Probabilidad Calculada')
    plt.ylim(0, 1)
    
    # Agregar etiquetas en las barras
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('experimento_estacionalidad.png', dpi=300)
    print("Gráfico guardado como 'experimento_estacionalidad.png'")

# ==========================================
# EXPERIMENTO 3: ROBUSTEZ ANTE DATOS FALTANTES
# ==========================================
def experimento_robustez():
    """
    Evalúa y compara cómo decae el Accuracy de la Red Bayesiana 
    vs la Red Neuronal cuando el usuario no ingresa todos los síntomas.
    """
    print("\n--- Ejecutando Experimento 3: Robustez (Bayesiana vs Neuronal) ---")
    
    # 1. Cargar datos asegurando que no haya trampa (Data Leakage)
    df = pd.read_csv("pacientes_sinteticos.csv")
    X = df.drop(columns=['Enfermedad'])
    y = df['Enfermedad']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Tomar 200 casos para que el experimento sea rápido
    X_sample = X_test.sample(200, random_state=42)
    y_sample = y_test.loc[X_sample.index].tolist()

    modelo_nn = joblib.load("modelo_neuronal.pkl")

    sintomas = [
        'Fiebre', 'Tos', 'Mialgia', 'Perdida_Olfato', 'Dolor_Cabeza',
        'Viaje_Reciente', 'Contacto_Positivo', 'Dolor_Retroocular',
        'Sarpullido', 'Diarrea'
    ]

    resultados = []

    # Vamos a probar dándole a la IA 10 síntomas, luego 8, 6, 4, y solo 2.
    for num_mantener in [10, 8, 6, 4, 2]:
        sintomas_activos = sintomas[:num_mantener]
        sintomas_ocultos = sintomas[num_mantener:]
        
        # --- A. Predicción Red Neuronal ---
        # A la red neuronal no le podemos "borrar" columnas, así que simulamos
        # que el usuario dejó el checkbox vacío (0)
        X_nn_ruido = X_sample.copy()
        for s in sintomas_ocultos:
            X_nn_ruido[s] = 0 
        
        pred_nn = modelo_nn.predict(X_nn_ruido)
        acc_nn = accuracy_score(y_sample, pred_nn)

        # --- B. Predicción Red Bayesiana ---
        pred_bayes = []
        for _, fila in X_sample.iterrows():
            # A la Bayesiana sí le podemos ocultar la variable del diccionario
            evidencia = {s: fila[s] for s in sintomas_activos}
            probs, _ = realizar_inferencia(evidencia)
            prediccion = max(probs, key=probs.get)
            pred_bayes.append(prediccion)
            
        acc_bayes = accuracy_score(y_sample, pred_bayes)

        resultados.append({
            'Sintomas Evaluados': num_mantener,
            'Red Neuronal': acc_nn * 100,
            'Red Bayesiana': acc_bayes * 100
        })

    df_resultados = pd.DataFrame(resultados)
    print("\nResultados de Exactitud (%) según síntomas visibles:")
    print(df_resultados.to_string(index=False))

    # Gráfico Comparativo
    plt.figure(figsize=(8, 5))
    plt.plot(df_resultados['Sintomas Evaluados'], df_resultados['Red Neuronal'], marker='o', linewidth=2.5, label='Red Neuronal', color='orange')
    plt.plot(df_resultados['Sintomas Evaluados'], df_resultados['Red Bayesiana'], marker='s', linewidth=2.5, label='Red Bayesiana', color='blue')
    
    plt.title('Caída de Exactitud por Falta de Información (Robustez)')
    plt.xlabel('Cantidad de Síntomas Ingresados por el Paciente')
    plt.ylabel('Exactitud Global (Accuracy %)')
    plt.xlim(10, 2) # Invertir eje X (de 10 a 2 síntomas)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('experimento_robustez.png', dpi=300)
    print("Gráfico guardado como 'experimento_robustez.png'")

if __name__ == "__main__":
    experimento_estacionalidad()
    experimento_robustez()