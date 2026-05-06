LEVANTAR APP
    Crear un entorno virtual: python -m venv .venv
    
    Activar el entorno: .venv\Scripts\activate
    
    python -m pip install --upgrade pip
    
    python -m pip install -r requirements.txt
    
    python -m streamlit run app.py   

FALTA:
-Agregar mas sintomas
-Agregar otra info visual
-agregar otras acciones recomendadas

COMPONENTES DEL SISTEMA EXPERTO:
    Base de Conocimiento (BC): Está representada por la estructura del modelo y las tablas CPD.
    Motor de Inferencia: Implementas VariableElimination, cumpliendo con el ciclo de emparejar y ejecutar de forma estocástica.  
    Subsistema de Explicación: Incluyes una salida de texto que justifica la clasificación, un requisito clave de los SE.  

Dinamicidad: La función cargar_probabilidades_2026 dota al agente de una Base de Conocimiento Dinámica, permitiendo que se actualice con datos epidemiológicos recientes.  

Manejo de Incertidumbre: A diferencia de un sistema de reglas simple (IF-THEN), tu red puede manejar casos donde los síntomas se solapan entre COVID-19 y Dengue, calculando la probabilidad más alta (Máximo a Posteriori).  

Integración de Datos Reales: El uso de pandas para calcular la prevalencia (Probabilidades a Priori) permite que el agente "aprenda" del entorno.
