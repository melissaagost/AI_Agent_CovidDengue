# Leer el CSV generado en el paso anterior y utilizar la librería scikit-learn (específicamente MLPClassifier) para entrenar la red neuronal.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

print("Cargando el dataset de pacientes sintéticos...")
# 1. Leer el archivo CSV generado
df = pd.read_csv("pacientes_sinteticos.csv")

# 2. Separar las variables de entrada (X: síntomas/contexto) de la etiqueta de salida (y: enfermedad)
X = df.drop(columns=['Enfermedad']) # Todo menos la columna enfermedad
y = df['Enfermedad']               # Únicamente la columna enfermedad

# 3. Dividir los datos: 80% para entrenar la red y 20% para evaluar qué tan bien aprendió
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Entrenando la Red Neuronal (MLPClassifier) con {len(X_train)} registros...")

# 4. Configurar la arquitectura de la Red Neuronal
# Usamos 2 capas ocultas (una de 16 neuronas y otra de 8 neuronas)
# 'max_iter=500' le da hasta 500 ciclos de entrenamiento para converger
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

print("¡Entrenamiento completado!")

# 5. Evaluar preliminarmente el modelo con el 20% de datos retenidos
y_pred = mlp.predict(X_test)
print("\nReporte preliminar de rendimiento de la Red Neuronal:")
print(classification_report(y_test, y_pred))

# 6. Exportar el modelo entrenado a un archivo físico usando joblib
modelo_filename = "modelo_neuronal.pkl"
joblib.dump(mlp, modelo_filename)
print(f"Modelo guardado con éxito como: '{modelo_filename}'")