#experimentos 2 y 3

def experimento_estacionalidad():
    """
    Evalúa cómo el factor estacional (verano: ×1.8)
    afecta el prior de Dengue en la Red Bayesiana.
    """
    from unittest.mock import patch
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter issues
    import matplotlib.pyplot as plt

    # Paciente de prueba
    evidencia_base = {
        'Fiebre': 1, 'Tos': 0, 'Mialgia': 1,
        'Perdida_Olfato': 0, 'Dolor_Cabeza': 1,
        'Viaje_Reciente': 1, 'Contacto_Positivo': 0,
        'Dolor_Retroocular': 1, 'Sarpullido': 1,
        'Diarrea': 1
    }

    resultados_estacion = {'Verano': [], 'Invierno': []}

    # Test en verano (dic-mar)
    import sys
    original_now = datetime.now

    for mes in [12, 1, 2, 3]:  # Verano
        with patch('agente_medico.datetime') as mock_dt:
            mock_dt.now.return_value.month = mes
            from agente_medico import realizar_inferencia
            probs, _ = realizar_inferencia(evidencia_base)
            resultados_estacion['Verano'].append(probs['Dengue'])

    for mes in [6, 7, 8, 9]:  # Invierno
        with patch('agente_medico.datetime') as mock_dt:
            mock_dt.now.return_value.month = mes
            from agente_medico import realizar_inferencia
            probs, _ = realizar_inferencia(evidencia_base)
            resultados_estacion['Invierno'].append(probs['Dengue'])

    # Comparar
    promedio_verano = sum(resultados_estacion['Verano']) / len(resultados_estacion['Verano'])
    promedio_invierno = sum(resultados_estacion['Invierno']) / len(resultados_estacion['Invierno'])

    print(f"P(Dengue) en Verano: {promedio_verano:.3f}")
    print(f"P(Dengue) en Invierno: {promedio_invierno:.3f}")
    print(f"Incremento: {(promedio_verano/promedio_invierno):.1f}x")

    # Generar gráfico
    plt.figure(figsize=(8, 5))
    plt.bar(['Verano', 'Invierno'], [promedio_verano, promedio_invierno], color=['red', 'blue'])
    plt.title('P(Dengue) según Factor Estacional')
    plt.ylabel('Probabilidad')
    plt.ylim(0, 1)
    plt.savefig('experimento_estacional.png', dpi=300)

    return {'verano': promedio_verano, 'invierno': promedio_invierno}





def experimento_robustez():
    """
    Evalúa cómo degrada el accuracy cuando faltan síntomas.
    Hipótesis: Accuracy decrece ~10% por cada 2-3 síntomas faltantes.
    """
    import itertools
    from agente_medico import realizar_inferencia
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend to avoid Tkinter issues
    import matplotlib.pyplot as plt

    # Cargar dataset de prueba
    df_prueba = pd.read_csv('pacientes_sinteticos.csv').sample(100, random_state=42)

    sintomas = ['Fiebre', 'Tos', 'Mialgia', 'Perdida_Olfato', 'Dolor_Cabeza',
                'Viaje_Reciente', 'Contacto_Positivo', 'Dolor_Retroocular',
                'Sarpullido', 'Diarrea']

    resultados_robustez = []

    # Test con diferentes cantidades de síntomas
    for num_sintomas in [1, 3, 5, 7, 10]:
        correctas = 0

        # Seleccionar subset aleatorio de síntomas
        sintomas_subset = sintomas[:num_sintomas]

        for _, fila in df_prueba.iterrows():
            evidencia = {s: fila[s] for s in sintomas_subset}
            probs, _ = realizar_inferencia(evidencia)
            prediccion = max(probs, key=probs.get)

            if prediccion == fila['Enfermedad']:
                correctas += 1

        accuracy = correctas / len(df_prueba)
        resultados_robustez.append({'num_sintomas': num_sintomas, 'accuracy': accuracy})

    df_resultados = pd.DataFrame(resultados_robustez)

    # Gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(df_resultados['num_sintomas'], df_resultados['accuracy'], marker='o', linewidth=2)
    plt.xlabel('Cantidad de Síntomas')
    plt.ylabel('Accuracy')
    plt.title('Robustez de Red Bayesiana ante Pérdida de Síntomas')
    plt.grid(True, alpha=0.3)
    plt.savefig('experimento_robustez.png', dpi=300)

    return df_resultados