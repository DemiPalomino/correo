# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

app = Flask(__name__)

# Cargar el dataset (asegúrate de que la ruta al archivo 'correo.csv' sea correcta)
df = pd.read_csv('correo.csv', encoding='latin-1')

# Ajustar el nombre de las columnas
X = df['Mensaje']  # Cambia a 'Mensaje' para acceder al texto del correo
y = df['Etiqueta']  # Cambia a 'Etiqueta' para acceder a las etiquetas

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizar los mensajes
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Crear el modelo de árbol de decisiones
model = DecisionTreeClassifier()
model.fit(X_train_vectors, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mensaje_usuario = request.form['mensaje']
    
    # Vectorizar el mensaje del usuario
    mensaje_vectorizado = vectorizer.transform([mensaje_usuario])
    
    # Hacer la predicción
    prediccion = model.predict(mensaje_vectorizado)

    # Mostrar el resultado
    resultado = "Spam" if prediccion[0] == "Spam" else "No Spam"  # Cambia a "Spam" en la condición
    return render_template('index.html', resultado=resultado, mensaje=mensaje_usuario)

if __name__ == '__main__':
    app.run(debug=True)
