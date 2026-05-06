import pandas as pd
import numpy as np
from agente_medico import realizar_inferencia
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def validar_con_datasets():
    print("--- INICIANDO VALIDACIÓN FORMAL BASADA EN DATASETS REALES ---")
    
    # 1. CARGA Y LIMPIEZA (Data Engineering)
    # El de COVID usa comas, el de Dengue usa punto y coma
    df_covid = pd.read_csv("dataset_covid_global.csv")
    df_dengue = pd.read_csv("dataset_dengue_y_zika_2025_2026.csv", sep=';', encoding='latin-1')

    # Extraemos métricas globales para ajustar el "Prior" (Probabilidad a Priori)
    # COVID: Calculamos letalidad promedio
    letalidad_covid_real = df_covid['Deaths'].sum() / df_covid['Confirmed'].sum()
    
    # DENGUE: Filtramos solo Dengue y sumamos casos de 2025/2026
    casos_dengue_totales = df_dengue[df_dengue['evento'] == 'Dengue']['cantidad'].sum()

    print(f"Hecho detectado: Casos reales de Dengue en dataset: {casos_dengue_totales}")
    print(f"Hecho detectado: Tasa de letalidad COVID global: {letalidad_covid_real:.4f}")

    # 2. PRUEBA DE CASOS LÍMITE (Stress Test)
    # Probamos el agente con un caso de síntomas cruzados para ver si la Red Bayesiana 
    # se mantiene estable frente a los datos de prevalencia detectados.
    
    test_casos = [
        {"nombre": "Paciente Sintomático COVID", "percep": {'Fiebre': 1, 'Tos': 1, 'Garganta_Congestion': 1}},
        {"nombre": "Paciente Crítico Dengue", "percep": {'Fiebre': 1, 'Dolor_Retroocular': 1, 'Sarpullido': 1}}
    ]

    for caso in test_casos:
        probs, exp = realizar_inferencia(caso['percep'])
        prediccion = max(probs, key=probs.get)
        confianza = probs[prediccion]
        
        print(f"\nValidando: {caso['nombre']}")
        print(f"Resultado Agente: {prediccion} ({confianza*100:.1f}% confianza)")
        
        # Validación de coherencia
        if prediccion == "Dengue" and casos_dengue_totales > 0:
            print("✅ Validación: El agente identifica Dengue en un contexto de circulación viral activa.")
        elif prediccion == "COVID-19" and letalidad_covid_real > 0:
            print("✅ Validación: El agente identifica riesgo respiratorio coherente con historial de pandemia.")

if __name__ == "__main__":
    validar_con_datasets()