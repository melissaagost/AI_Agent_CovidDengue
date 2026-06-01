import pandas as pd
import numpy as np

# Configuramos una semilla aleatoria para que el grupo obtenga siempre los mismos datos
np.random.seed(42)

# 1. Definimos los parámetros generales
NUM_PACIENTES = 5000
ENFERMEDADES = ['Ninguna', 'Dengue', 'COVID-19']

# Probabilidades a Priori Base (Contexto epidemiológico normal)
PRIOR_BASE = {
    'Ninguna': 0.60,
    'Dengue': 0.15,
    'COVID-19': 0.25
}

# 2. Matrices de Probabilidad Condicional (CPD) basadas en su modelo
# P(Sintoma = 1 | Enfermedad)
TABLA_CPD = {
    'Fiebre':           {'Ninguna': 0.05, 'Dengue': 0.90, 'COVID-19': 0.80},
    'Tos':              {'Ninguna': 0.10, 'Dengue': 0.05, 'COVID-19': 0.75},
    'Mialgia':          {'Ninguna': 0.08, 'Dengue': 0.85, 'COVID-19': 0.40},
    'Perdida_Olfato':   {'Ninguna': 0.01, 'Dengue': 0.02, 'COVID-19': 0.70},
    'Dolor_Cabeza':     {'Ninguna': 0.12, 'Dengue': 0.80, 'COVID-19': 0.50},
    'Viaje_Reciente':   {'Ninguna': 0.05, 'Dengue': 0.45, 'COVID-19': 0.20},
    'Contacto_Positivo': {'Ninguna': 0.02, 'Dengue': 0.40, 'COVID-19': 0.35},
    'Dolor_Retroocular': {'Ninguna': 0.01, 'Dengue': 0.40, 'COVID-19': 0.05},
    'Sarpullido':       {'Ninguna': 0.02, 'Dengue': 0.55, 'COVID-19': 0.01},
    'Diarrea':          {'Ninguna': 0.05, 'Dengue': 0.50, 'COVID-19': 0.30}
}

print(f"Generando dataset sintético de {NUM_PACIENTES} pacientes...")

lista_pacientes = []

for i in range(NUM_PACIENTES):
    # a. Determinar factor estacional para este paciente de forma aleatoria (50% probabilidad de que sea Verano)
    es_verano = np.random.choice([1, 0], p=[0.5, 0.5])
    
    # b. Ajustar probabilidades a priori según la regla de negocio (Verano multiplica Dengue x 1.8)
    prior_actual = PRIOR_BASE.copy()
    if es_verano == 1:
        prior_actual['Dengue'] = prior_actual['Dengue'] * 1.8
        
    # Normalizar los priors para que la suma vuelva a ser 1.0
    suma_priors = sum(prior_actual.values())
    priors_normalizados = [prior_actual[enfermedad] / suma_priors for enfermedad in ENFERMEDADES]
    
    # c. Asignar la enfermedad real del paciente basándonos en los priors ajustados
    enfermedad_asignada = np.random.choice(ENFERMEDADES, p=priors_normalizados)
    
    # d. Generar los síntomas (1 o 0) en base a la enfermedad asignada (Lógica CPD)
    datos_paciente = {
        'Verano': es_verano
    }
    
    for sintoma, probs in TABLA_CPD.items():
        prob_positivo = probs[enfermedad_asignada]
        # Evaluamos el síntoma como 1 (presente) o 0 (ausente)
        datos_paciente[sintoma] = np.random.choice([1, 0], p=[prob_positivo, 1 - prob_positivo])
        
    # e. Guardar el diagnóstico final (la etiqueta que aprenderá la Red Neuronal)
    datos_paciente['Enfermedad'] = enfermedad_asignada
    
    lista_pacientes.append(datos_paciente)

# 3. Convertir a DataFrame de Pandas y exportar a CSV
df_pacientes = pd.DataFrame(lista_pacientes)

# Reordenar columnas de forma prolija
columnas = ['Verano', 'Fiebre', 'Tos', 'Mialgia', 'Perdida_Olfato', 'Dolor_Cabeza', 
            'Viaje_Reciente', 'Contacto_Positivo', 'Dolor_Retroocular', 'Sarpullido', 
            'Diarrea', 'Enfermedad']
df_pacientes = df_pacientes[columnas]

# Guardar archivo físico
csv_filename = "pacientes_sinteticos.csv"
df_pacientes.to_csv(csv_filename, index=False)

print(f"Dataset generado")
print(f"Archivo guardado como: '{csv_filename}'")
print("\nDistribución de diagnósticos generados:")
print(df_pacientes['Enfermedad'].value_counts())
print("\nMuestra de los primeros 5 pacientes:")
print(df_pacientes.head())