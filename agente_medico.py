"""
agente_medico.py
Motor de Inferencia Bayesiano para clasificación de riesgo Dengue / COVID-19.
Sistema Experto de Triaje Infectológico — Corrientes, Argentina.

Los priors se calculan a partir de datasets reales:
  - dataset_covid_global.csv            → tasa de letalidad global COVID
  - dataset_dengue_y_zika_2025_2026.csv → circulación de Dengue en Corrientes 2025
"""

import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. PROBABILIDADES A PRIORI — calculadas desde datasets reales
# ---------------------------------------------------------------------------

_PRIOR_NINGUNA = 0.65   # fallback si los CSVs no se pueden leer
_PRIOR_DENGUE  = 0.15
_PRIOR_COVID   = 0.20

_DIR = os.path.dirname(os.path.abspath(__file__))


def _cargar_prior_covid() -> float:
    """
    Usa la tasa de letalidad global del CSV de COVID como señal de
    peso relativo de la enfermedad en el prior.
    Tasa real ≈ 3.97%  →  mapeada a un prior entre 0.10 y 0.35.
    """
    path = os.path.join(_DIR, "dataset_covid_global.csv")
    df = pd.read_csv(path)
    letalidad = df['Deaths'].sum() / df['Confirmed'].sum()
    # Mapeo lineal: letalidad 0–10% → prior 0.10–0.35
    prior = 0.10 + (letalidad / 0.10) * 0.25
    return float(min(max(prior, 0.10), 0.35))


def _cargar_prior_dengue_corrientes() -> float:
    """
    Calcula la proporción de casos de Dengue registrados en Corrientes
    sobre el total nacional (dataset SNVS 2025) y la usa para escalar
    el prior base de Dengue.
    """
    path = os.path.join(_DIR, "dataset_dengue_y_zika_2025_2026.csv")
    df = pd.read_csv(path, sep=';', encoding='latin-1')
    df_dengue       = df[df['evento'] == 'Dengue']
    total_nacional  = df_dengue['cantidad'].sum()
    casos_corrientes = df_dengue[
        df_dengue['provincia_residencia'].str.lower() == 'corrientes'
    ]['cantidad'].sum()

    if total_nacional == 0:
        return _PRIOR_DENGUE

    # Proporción local ≈ 54 / 17576 ≈ 0.003
    # Factor ×10 para dar peso epidemiológico local razonable al prior
    proporcion = casos_corrientes / total_nacional
    prior = 0.15 + proporcion * 10
    return float(min(max(prior, 0.05), 0.40))


def calcular_prevalencia_estacional() -> list[float]:
    """
    Calcula las probabilidades a priori [Ninguna, Dengue, COVID-19]
    combinando:
      1. Datos reales de los datasets (COVID global, Dengue Corrientes).
      2. Factor estacional: verano austral (dic–mar) sube el prior de Dengue.

    Si los CSVs no están disponibles, usa valores de fallback.
    Retorna lista normalizada que suma 1.0.
    """
    try:
        p_covid  = _cargar_prior_covid()
        p_dengue = _cargar_prior_dengue_corrientes()
        p_ninguna = max(1.0 - p_covid - p_dengue, 0.05)
    except Exception as e:
        print(f"[agente_medico] Advertencia: no se pudieron leer los datasets ({e}). "
              "Usando priors por defecto.")
        p_covid   = _PRIOR_COVID
        p_dengue  = _PRIOR_DENGUE
        p_ninguna = _PRIOR_NINGUNA

    # Factor estacional: verano austral aumenta circulación de Dengue
    mes_actual = datetime.now().month
    if mes_actual in [12, 1, 2, 3]:
        p_dengue  *= 1.8
        p_ninguna *= 0.85

    # Normalización garantizada
    total = p_ninguna + p_dengue + p_covid
    return [p_ninguna / total, p_dengue / total, p_covid / total]


# ---------------------------------------------------------------------------
# 2. CONSTRUCCIÓN DE LA RED BAYESIANA
# ---------------------------------------------------------------------------

modelo = DiscreteBayesianNetwork([
    ('Enfermedad', 'Fiebre'),
    ('Enfermedad', 'Tos'),
    ('Enfermedad', 'Mialgia'),
    ('Enfermedad', 'Viaje_Reciente'),
    ('Enfermedad', 'Contacto_Dengue'),
    ('Enfermedad', 'Dolor_Retroocular'),
    ('Enfermedad', 'Perdida_Olfato'),
    ('Enfermedad', 'Sarpullido'),
    ('Enfermedad', 'Garganta_Congestion'),
])

# Nodo raíz: Enfermedad — [Ninguna=0, Dengue=1, COVID-19=2]
prevalencia = calcular_prevalencia_estacional()
cpd_enf = TabularCPD(
    variable='Enfermedad', variable_card=3,
    values=[[prevalencia[0]], [prevalencia[1]], [prevalencia[2]]]
)

# ---------------------------------------------------------------------------
# CPDs para síntomas (filas: [P(NO), P(SÍ)] / columnas: [Ninguna, Dengue, COVID])
# ---------------------------------------------------------------------------

cpd_fiebre = TabularCPD(
    variable='Fiebre', variable_card=2,
    values=[[0.95, 0.10, 0.20],
            [0.05, 0.90, 0.80]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_tos = TabularCPD(
    variable='Tos', variable_card=2,
    values=[[0.98, 0.80, 0.10],
            [0.02, 0.20, 0.90]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_mialgia = TabularCPD(
    variable='Mialgia', variable_card=2,
    values=[[0.99, 0.05, 0.70],
            [0.01, 0.95, 0.30]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_retroocular = TabularCPD(
    variable='Dolor_Retroocular', variable_card=2,
    values=[[0.99, 0.20, 0.95],
            [0.01, 0.80, 0.05]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_olfato = TabularCPD(
    variable='Perdida_Olfato', variable_card=2,
    values=[[0.99, 0.98, 0.30],
            [0.01, 0.02, 0.70]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_sarpullido = TabularCPD(
    variable='Sarpullido', variable_card=2,
    values=[[0.98, 0.45, 0.99],
            [0.02, 0.55, 0.01]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_garganta = TabularCPD(
    variable='Garganta_Congestion', variable_card=2,
    values=[[0.95, 0.90, 0.30],
            [0.05, 0.10, 0.70]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_viaje = TabularCPD(
    variable='Viaje_Reciente', variable_card=2,
    values=[[0.99, 0.20, 0.90],
            [0.01, 0.80, 0.10]],
    evidence=['Enfermedad'], evidence_card=[3]
)

cpd_contacto = TabularCPD(
    variable='Contacto_Dengue', variable_card=2,
    values=[[0.99, 0.10, 0.95],
            [0.01, 0.90, 0.05]],
    evidence=['Enfermedad'], evidence_card=[3]
)

modelo.add_cpds(
    cpd_enf, cpd_fiebre, cpd_tos, cpd_mialgia,
    cpd_viaje, cpd_contacto, cpd_retroocular,
    cpd_olfato, cpd_sarpullido, cpd_garganta
)

assert modelo.check_model(), "La Red Bayesiana tiene inconsistencias en sus CPDs."

inferencia = VariableElimination(modelo)

# ---------------------------------------------------------------------------
# 3. MOTOR DE INFERENCIA
# ---------------------------------------------------------------------------

ETIQUETAS = ["Ninguna", "Dengue", "COVID-19"]


def realizar_inferencia(percepciones: dict) -> tuple[dict, str]:
    """
    Ejecuta la inferencia bayesiana dados los síntomas/factores observados.

    Parámetros
    ----------
    percepciones : dict
        Evidencia binaria {variable: 0|1}.
        Variables admitidas: 'Fiebre', 'Tos', 'Mialgia', 'Viaje_Reciente',
        'Contacto_Dengue', 'Dolor_Retroocular', 'Perdida_Olfato',
        'Sarpullido', 'Garganta_Congestion'.

    Retorna
    -------
    probabilidades : dict
        {etiqueta: probabilidad} para Ninguna, Dengue, COVID-19.
    explicacion : str
        Justificación textual del diagnóstico sugerido.
    """
    evidencia_activa = {k: v for k, v in percepciones.items() if v in (0, 1)}

    resultado      = inferencia.query(variables=['Enfermedad'], evidence=evidencia_activa)
    probabilidades = resultado.values   # array [Ninguna, Dengue, COVID-19]

    idx_max    = probabilidades.argmax()
    prediccion = ETIQUETAS[idx_max]
    confianza  = probabilidades[idx_max] * 100

    # --- Subsistema de Explicación ---
    explicacion = (
        f"Diagnóstico sugerido: **{prediccion}** con una probabilidad del "
        f"**{confianza:.1f}%**.\n\nJustificación del motor de inferencia:\n"
    )

    if prediccion == "Dengue":
        if percepciones.get('Viaje_Reciente') == 1:
            explicacion += "- El antecedente de viaje reciente aumenta significativamente la probabilidad.\n"
        if percepciones.get('Contacto_Dengue') == 1:
            explicacion += "- El contacto con un paciente positivo es un factor de riesgo determinante.\n"
        if percepciones.get('Dolor_Retroocular') == 1:
            explicacion += "- El dolor retroocular es un síntoma clásico del Dengue que refuerza el diagnóstico.\n"
        if percepciones.get('Sarpullido') == 1:
            explicacion += "- La presencia de sarpullido/erupción cutánea es un signo dermatológico fuerte para Dengue.\n"
        if percepciones.get('Mialgia') == 1:
            explicacion += "- Las mialgias intensas son características del cuadro febril del Dengue.\n"
        if datetime.now().month in [12, 1, 2, 3]:
            explicacion += "- Factor estacional aplicado: alta prevalencia de Dengue en verano austral en Corrientes.\n"
        explicacion += "- Prior ajustado con datos reales de circulación de Dengue en Corrientes 2025 (SNVS).\n"

    elif prediccion == "COVID-19":
        if percepciones.get('Tos') == 1:
            explicacion += "- La presencia de tos orienta el diagnóstico hacia patologías respiratorias.\n"
        if percepciones.get('Perdida_Olfato') == 1:
            explicacion += "- La pérdida de olfato/gusto es un marcador altamente específico de COVID-19.\n"
        if percepciones.get('Garganta_Congestion') == 1:
            explicacion += "- Los síntomas de vías respiratorias superiores orientan hacia COVID-19.\n"
        explicacion += "- Prior ajustado con tasa de letalidad global real del dataset COVID (≈3.97%).\n"

    else:
        explicacion += "- El perfil sintomático no presenta indicadores críticos de brote infeccioso conocido.\n"
        explicacion += "- Se recomienda seguimiento clínico estándar.\n"

    return dict(zip(ETIQUETAS, probabilidades)), explicacion
