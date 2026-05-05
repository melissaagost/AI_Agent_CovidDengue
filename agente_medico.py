import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def cargar_probabilidades_2026():
    # Carga del primer dataset
    df_dengue = pd.read_csv('dataset dengue y zika 2025 2026.csv', encoding='latin-1') 
    
    # CORRECCIÓN: Si el .xls es en realidad un CSV disfrazado
    try:
        # Intentamos leerlo como CSV dado el error de 'found b'Country/''
        df_covid = pd.read_csv('dataset covid global.xls', encoding='latin-1')
    except Exception:
        # Si falla, volvemos al motor de Excel por si acaso
        df_covid = pd.read_excel('dataset covid global.xls', engine='xlrd')
    
    # Cálculo de Prevalencia (Probabilidades a Priori)
    total_casos = len(df_dengue) + len(df_covid) + 1000  # 1000 como base de casos sanos
    p_dengue = len(df_dengue) / total_casos
    p_covid = len(df_covid) / total_casos
    p_ninguna = 1 - (p_dengue + p_covid)
    
    return [p_ninguna, p_dengue, p_covid]

# 1. Estructura de la Red (PAMA: Medios-Fines) 
modelo = DiscreteBayesianNetwork([
    ('Enfermedad', 'Fiebre'), 
    ('Enfermedad', 'Tos'),
    ('Enfermedad', 'Mialgia')
    #agregar mas sintomas
])

# 2. Base de Conocimiento Dinámica 
prevalencia = cargar_probabilidades_2026()
cpd_enf = TabularCPD(variable='Enfermedad', variable_card=3, 
                     values=[[prevalencia[0]], [prevalencia[1]], [prevalencia[2]]])

# Definición de CPDs (Probabilidades Condicionales P(H|E)) 
# Los valores internos pueden refinarse con un análisis de frecuencia en los CSV
cpd_fiebre = TabularCPD(variable='Fiebre', variable_card=2,
                        values=[[0.95, 0.1, 0.2], [0.05, 0.9, 0.8]],
                        evidence=['Enfermedad'], evidence_card=[3])

cpd_tos = TabularCPD(variable='Tos', variable_card=2,
                     values=[[0.98, 0.8, 0.1], [0.02, 0.2, 0.9]],
                     evidence=['Enfermedad'], evidence_card=[3])

cpd_mialgia = TabularCPD(variable='Mialgia', variable_card=2,
                        values=[[0.99, 0.05, 0.7], [0.01, 0.95, 0.3]],
                        evidence=['Enfermedad'], evidence_card=[3])

# 3. Registro y Motor de Inferencia 
modelo.add_cpds(cpd_enf, cpd_fiebre, cpd_tos, cpd_mialgia)
inferencia = VariableElimination(modelo)

def realizar_inferencia(percepciones):

    # Proceso de razonamiento del Agente
    resultado = inferencia.query(variables=['Enfermedad'], evidence=percepciones)
    probabilidades = resultado.values
    etiquetas = ["Ninguna", "Dengue", "COVID-19"]
    
    # Subsistema de Explicación
    idx_max = probabilidades.argmax()
    prediccion = etiquetas[idx_max]
    explicacion = f"El sistema clasifica al paciente como {prediccion} basándose en datos epidemiológicos de 2026."
    
    return dict(zip(etiquetas, probabilidades)), explicacion