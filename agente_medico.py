import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from datetime import datetime

def calcular_prevalencia_estacional():
    # Simulamos el bloque de lectura de CSV/XLS que ya tenías
    # Para el ejemplo, usaremos probabilidades base
    p_dengue_base = 0.15
    p_covid_base = 0.20
    p_ninguna_base = 0.65
    
    # 1. INCORPORACIÓN DEL FACTOR ESTACIONAL
    # Se evalúa si es verano en Corrientes para ajustar la probabilidad a priori
    mes_actual = datetime.now().month
    if mes_actual in [12, 1, 2, 3]: # Meses de verano
        p_dengue_base *= 1.8 # Aumenta la prevalencia de Dengue
        
    # Normalizamos para que la suma de las 3 probabilidades sea 1.0
    total = p_ninguna_base + p_dengue_base + p_covid_base
    return [p_ninguna_base/total, p_dengue_base/total, p_covid_base/total]

# 2. EXPANSIÓN DE LA RED BAYESIANA
# Se añaden nodos para antecedentes epidemiológicos
modelo = DiscreteBayesianNetwork([
    ('Enfermedad', 'Fiebre'), 
    ('Enfermedad', 'Tos'),
    ('Enfermedad', 'Mialgia'),
    ('Enfermedad', 'Viaje_Reciente'),      # Nuevo nodo
    ('Enfermedad', 'Contacto_Dengue')    # Nuevo nodo
])

prevalencia = calcular_prevalencia_estacional()
cpd_enf = TabularCPD(variable='Enfermedad', variable_card=3, 
                     values=[[prevalencia[0]], [prevalencia[1]], [prevalencia[2]]])

# Definición de CPDs para síntomas (0: No, 1: Sí)
cpd_fiebre = TabularCPD(variable='Fiebre', variable_card=2,
                        values=[[0.95, 0.1, 0.2], [0.05, 0.9, 0.8]],
                        evidence=['Enfermedad'], evidence_card=[3])

cpd_tos = TabularCPD(variable='Tos', variable_card=2,
                     values=[[0.98, 0.8, 0.1], [0.02, 0.2, 0.9]],
                     evidence=['Enfermedad'], evidence_card=[3])

cpd_mialgia = TabularCPD(variable='Mialgia', variable_card=2,
                         values=[[0.99, 0.05, 0.7], [0.01, 0.95, 0.3]],
                         evidence=['Enfermedad'], evidence_card=[3])

# Definición de CPDs para factores de riesgo
cpd_viaje = TabularCPD(variable='Viaje_Reciente', variable_card=2,
                       values=[[0.99, 0.2, 0.9],   # Prob de NO haber viajado dado (Ninguna, Dengue, Covid)
                               [0.01, 0.8, 0.1]],  # Prob de SÍ haber viajado
                       evidence=['Enfermedad'], evidence_card=[3])

cpd_contacto = TabularCPD(variable='Contacto_Dengue', variable_card=2,
                          values=[[0.99, 0.1, 0.95], 
                                  [0.01, 0.9, 0.05]], 
                          evidence=['Enfermedad'], evidence_card=[3])

modelo.add_cpds(cpd_enf, cpd_fiebre, cpd_tos, cpd_mialgia, cpd_viaje, cpd_contacto)
inferencia = VariableElimination(modelo)

def realizar_inferencia(percepciones):
    resultado = inferencia.query(variables=['Enfermedad'], evidence=percepciones)
    probabilidades = resultado.values
    etiquetas = ["Ninguna", "Dengue", "COVID-19"]
    
    idx_max = probabilidades.argmax()
    prediccion = etiquetas[idx_max]
    confianza = probabilidades[idx_max] * 100
    
    # 3. MEJORA DEL SUBSISTEMA DE EXPLICACIÓN
    # Justifica el "por qué" de cada conclusión
    explicacion = f"Diagnóstico sugerido: {prediccion} con una probabilidad del {confianza:.1f}%.\n"
    explicacion += "Justificación del motor de inferencia:\n"
    
    if prediccion == "Dengue":
        if percepciones.get('Viaje_Reciente') == 1:
            explicacion += "- El antecedente de viaje reciente aumenta significativamente la probabilidad.\n"
        if percepciones.get('Contacto_Dengue') == 1:
            explicacion += "- El contacto con un paciente positivo es un factor de riesgo determinante.\n"
        if datetime.now().month in [12, 1, 2, 3]:
            explicacion += "- Se aplicó un factor de riesgo por prevalencia alta de Dengue durante los meses de verano en Corrientes.\n"
            
    elif prediccion == "COVID-19":
        if percepciones.get('Tos') == 1:
            explicacion += "- La presencia de tos orienta el diagnóstico hacia patologías respiratorias.\n"
            
    return dict(zip(etiquetas, probabilidades)), explicacion

# Ejemplo de prueba (Percepción del agente)
# Paciente con fiebre, mialgia, que viajó a Brasil y tuvo contacto con Dengue
percepciones_actuales = {'Fiebre': 1, 'Mialgia': 1, 'Tos': 0, 'Viaje_Reciente': 1, 'Contacto_Dengue': 1}
prob, exp = realizar_inferencia(percepciones_actuales)
print(exp)