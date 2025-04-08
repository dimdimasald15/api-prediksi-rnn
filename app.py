from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf

from train_model import train_bp  # ✅ import blueprint

app = Flask(__name__)
app.register_blueprint(train_bp)  # ✅ daftarkan blueprint

# Load model
model = tf.keras.models.load_model('model_rnn_konsumsi.keras')

@app.route('/')
def index():
    return "API Prediksi Konsumsi Listrik RNN"

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
        return {"message": "Model retrained and saved successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
