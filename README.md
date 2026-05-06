  🚀 Ejecución del Proyecto

Pasos para configurar el entorno y levantar la aplicación:

    python -m venv .venv
    .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
    python -m streamlit run app.py
    python -m streamlit run app2.py


🧠 Arquitectura del Sistema Experto

    Base de Conocimiento (BC): Representada mediante la estructura de la Red Bayesiana y las tablas de Probabilidad Condicional (CPD) que almacenan el saber médico especializado.  

    Motor de Inferencia: Implementa el algoritmo VariableElimination, lo que permite realizar un razonamiento estocástico para deducir conclusiones bajo incertidumbre, característico de los Agentes Racionales que operan en entornos parcialmente observables. 

    Subsistema de Explicación: Genera justificaciones textuales sobre el diagnóstico sugerido, cumpliendo con el requisito de transparencia y auditabilidad del sistema.  

✨ Características Técnicas

    Manejo de Incertidumbre: A diferencia de los sistemas de reglas rígidos (IF-THEN), este modelo utiliza el Teorema de Bayes para calcular la probabilidad más alta (Máximo a Posteriori) cuando los síntomas se solapan entre enfermedades:
      
    P(A∣B)=P(B)P(B∣A)×P(A)​

    Dinamicidad: Gracias a la función cargar_probabilidades_2026, el agente posee una BC dinámica que se actualiza con datos epidemiológicos recientes.  

    Integración de Datos: El uso de la librería pandas permite procesar datasets reales para calcular prevalencias (probabilidades a priori), logrando que el agente se adapte a su entorno.  

📋 Diseño Formal (Descriptor REAS)

    Rendimiento: Clasificación con 90% de precisión y minimización de falsos negativos.

    Entorno: Centro de emergencias de alta afluencia (Corrientes, AR).

    Actuadores: Recomendación de derivación (Aislamiento/Sala General).

    Sensores: Interfaz de síntomas y antecedentes epidemiológicos.

📝 Próximos Pasos 

    [ ] Ampliar el catálogo de síntomas y factores de riesgo en la Red Bayesiana.

    [ ] Incorporar visualizaciones estadísticas avanzadas en la interfaz.

    [ ] Validar formalmente la precisión del modelo probabilístico frente a casos de prueba.

    [ ] Extender las acciones recomendadas según protocolos médicos de aislamiento actualizados.
