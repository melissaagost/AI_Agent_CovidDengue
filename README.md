# Sistema de Triaje


## EXPERIMENTOS
### Experimento 1: Diagnóstico bajo Incertidumbre
Status: Completado en generar_metricas.py
### Experimento 2: Cambios Estacionales
Status: Completado en experimentos.py
### Experimento 3: Robustez ante Pérdida de Datos
Status: Completado en experimentos.py


## VERIFICACIÓN MÓDULOS
### Check Bayesian Network
python -c "from agente_medico import realizar_inferencia; print('✓ Bayesiana OK')"

#### Check Dataset
python -c "import pandas as pd; df=pd.read_csv('pacientes_sinteticos.csv'); print(f'✓ Dataset: {len(df)} records')"

### Check Neural Network Model
python -c "import joblib; m=joblib.load('modelo_neuronal.pkl'); print('✓ Neuronal Model OK')"

### Quick Inference Test
python -c "from agente_medico import realizar_inferencia; p,_=realizar_inferencia({'Fiebre':1,'Tos':1,'Perdida_Olfato':1,'Mialgia':0,'Dolor_Cabeza':0,'Viaje_Reciente':0,'Contacto_Positivo':0,'Dolor_Retroocular':0,'Sarpullido':0,'Diarrea':0}); print(f'Diagnosis: {max(p, key=p.get)}')"

 ## EVALUAR PERFORMANCE DE LOS MODELOS
 ### Run metrics generation
python generar_metricas.py

Salidas:
    Confusion matrices (PNG files)
    Accuracy scores for both models
    Classification reports
    Stored in metricas_modelos.json

## PREGUNTAS FRECUENTES PARA LA DEFENSA

**P1:** "¿Cómo generan los priors de la Red Bayesiana?"
**R:** "Extraemos datos reales de los datasets SNVS y COVID global, calculamos tasa de letalidad y proporción de casos en Corrientes, e integramos factor estacional (×1.8 en verano austral)."

**P2:** "¿Por qué comparan Red Bayesiana con Red Neuronal?"
**R:** "Porque la Bayesiana es interpretable (cumple requerimientos regulatorios médicos) pero la Neuronal captura patrones no lineales. Usarlas en paralelo reduce errores de clasificación."

**P3:** "¿Cómo validan que 10 síntomas es suficiente?"
**R:** "A través del Experimento 3 (Robustez ante Pérdida de Datos), demostramos que con 10 síntomas conseguimos ~90% accuracy y que agregar más incrementa complejidad sin ganancia."

**P4:** "¿Qué harían en un entorno de producción?"
**R:** "Validaríamos prospectivamente con datos reales de Corrientes, monitorizaríamos drift de datos, actualizaríamos priors mensuales según SNVS, e integraríamos con la HCE existente."

