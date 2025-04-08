from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf

from train_model import train_bp, train_and_save_model

app = Flask(__name__)
app.register_blueprint(train_bp)

model = None
model_path = 'model_rnn_konsumsi.keras'

if os.path.exists(model_path):
    load_model()

def load_model():
    global model
    model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    status = "tersedia" if model else "tidak tersedia"
    return f"API Prediksi Konsumsi Listrik RNN (Model {status})"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['data']).reshape(1, len(data['data']), 1)
    prediction = model.predict(input_data)
    return jsonify({'prediksi': float(prediction[0][0])})

@app.route('/train_model', methods=['POST'])
def train():
    try:
        train_and_save_model()
        load_model()
        return {"message": "Model retrained and saved successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
