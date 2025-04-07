from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_rnn_konsumsi.keras')
scaler_y = joblib.load('scaler_y.save')

@app.route('/')
def index():
    return "API Prediksi Konsumsi Listrik RNN"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # ex: [108, 40, 40]
    
    # Ambil nilai terakhir sebagai input prediksi
    input_data = np.array(data[-1:]).reshape(1, 1, 1)
    
    # Lakukan scaling (optional, kalau kamu ingin juga simpan scaler_X)
    # Tapi kalau input sudah distandardkan manual, bisa langsung saja:
    
    pred_scaled = model.predict(input_data)
    
    # Inverse scaling ke bentuk asli
    pred_original = scaler_y.inverse_transform(pred_scaled)
    
    return jsonify({'prediksi': float(pred_original[0][0])})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
