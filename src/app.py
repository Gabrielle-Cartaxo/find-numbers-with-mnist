from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

app = Flask(__name__)

# Carregar ambos os modelos
model_linear = load_model('modelos/linear.h5')
model_cnn = load_model('modelos/cnn.h5')

def preprocess_image(img):
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array

@app.route('/predict_linear', methods=['POST'])
def predict_linear():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Converter para escala de cinza
    img_array = preprocess_image(img).reshape(1, 28 * 28)  # Remodelar para vetor linear
    prediction = model_linear.predict(img_array)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Converter para escala de cinza
    img_array = preprocess_image(img).reshape(1, 28, 28, 1)  # Manter as dimensões para CNN
    prediction = model_cnn.predict(img_array)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

@app.route('/')
def index():
    return '''
    <h1>MNIST Digit Prediction</h1>
    <form action="/predict_linear" method="post" enctype="multipart/form-data">
        <h2>Linear Model</h2>
        <input type="file" name="file"><br><br>
        <input type="submit" value="Predict">
    </form>
    <form action="/predict_cnn" method="post" enctype="multipart/form-data">
        <h2>CNN Model</h2>
        <input type="file" name="file"><br><br>
        <input type="submit" value="Predict">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
