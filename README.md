#Ejecución del Proyecto
Pasos para configurar el entorno y levantar la aplicación:

    python -m venv .venv
    .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
    python -m streamlit run app.py
    python -m streamlit run app2.py


# Arquitectura del Sistema Experto

## 1. Probabilidades a Priori (Priors)

El conocimiento *a priori* es aquel que es independiente de la experiencia o evidencia actual.  
En el agente, representan la probabilidad de que una persona tenga Dengue o COVID-19 antes de evaluar síntomas.

### Dinámica con los Datasets

El sistema no utiliza valores estáticos, sino que extrae conocimiento de datos reales para ajustarse al contexto de Corrientes.

#### Dataset COVID
- Calcula la tasa de letalidad global.
- Si la letalidad aumenta, el sistema incrementa preventivamente el peso de esta enfermedad en el diagnóstico.

#### Dataset Dengue
- Analiza la proporción de casos en Corrientes respecto al total nacional (SNVS 2025).
- Esto aporta una “sensibilidad local” al agente.

#### Factor Estacional
El sistema aplica una regla de negocio:

- Si detecta verano austral (`diciembre → marzo`), multiplica la probabilidad de Dengue por `1.8`.
- Esto simula el comportamiento epidemiológico real.

---

## 2. Red Bayesiana

La Red Bayesiana es un modelo gráfico que representa dependencias entre variables.  
En el proyecto se implementa utilizando la librería `pgmpy`.

### Estructura:

#### Nodo Raíz: 
  -La variable "Enfermedad" (que puede ser Ninguna, Dengue o COVID-19).  
#### Nodos Dependientes (Síntomas): "Fiebre", "Tos", "Mialgia", etc. Estos dependen directamente de la enfermedad.  

#### Tablas de Probabilidad Condicional (CPD): 
  -Son matrices que definen la relación lógica. Por ejemplo, la probabilidad de tener "Pérdida de Olfato" dado que el paciente tiene "COVID-19" es del 70%, mientras que para "Dengue" es solo del 2%. Esto permite al sistema distinguir entre patologías con síntomas similares.  


## 3. Motor de Inferencia

El motor de inferencia es el componente encargado de decidir qué cálculos aplicar para generar el diagnóstico probabilístico.

### Algoritmo: 
  -Utiliza VariableElimination para calcular la probabilidad a posteriori.  

### Funcionamiento: 
  -Toma la Memoria de Trabajo (los síntomas que el usuario marca en Streamlit) y los cruza con la Base de Conocimiento (la Red Bayesiana).  

### Resultado: 
  -Calcula cuál es la patología con mayor probabilidad una vez observada la evidencia (los síntomas).


# Componentes del SE y su implementación en código

#### Base de Conocimiento
  -La Red Bayesiana y sus tablas CPD (conocimiento permanente).  
#### Memoria de Trabajo
  -El diccionario evidencia con los síntomas actuales del paciente.  
#### Motor de Inferencias
  -El objeto VariableElimination que realiza el cálculo probabilístico.  
#### Subsistema de Explicación
  -La función realizar_inferencia genera un texto justificando el diagnóstico según los síntomas detectados (obligatorio en SE). 
#### Interfaz de Usuario
  -La aplicación de Streamlit (app.py) que actúa como sensor (entrada) y actuador (salida).  


# Características Técnicas

    Manejo de Incertidumbre: A diferencia de los sistemas de reglas rígidos (IF-THEN), este modelo utiliza el Teorema de Bayes para calcular la probabilidad más alta (Máximo a Posteriori) cuando los síntomas se solapan entre enfermedades:
      
    P(A∣B)=P(B)P(B∣A)×P(A)​

    Dinamicidad: Gracias a la función cargar_probabilidades_2026, el agente posee una BC dinámica que se actualiza con datos epidemiológicos recientes.  

    Integración de Datos: El uso de la librería pandas permite procesar datasets reales para calcular prevalencias (probabilidades a priori), logrando que el agente se adapte a su entorno.  

# Diseño Formal (Descriptor REAS)

    Rendimiento: Clasificación con 90% de precisión y minimización de falsos negativos.

    Entorno: Centro de emergencias de alta afluencia (Corrientes, AR).

    Actuadores: Recomendación de derivación (Aislamiento/Sala General).

    Sensores: Interfaz de síntomas y antecedentes epidemiológicos.
