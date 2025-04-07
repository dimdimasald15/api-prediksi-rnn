from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler


app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model_rnn_konsumsi.keras')
# Load scaler
# scaler = joblib.load('scaler.pkl')
# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

@app.route('/')
def index():
    return "API Prediksi Konsumsi Listrik RNN"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Data input harus dalam format array: [val1, val2, val3]
    input_data = np.array(data['data'])
    input_scaled = scaler_X.transform(input_data)
    input_scaled = input_scaled.reshape(1, len(data['data']), 1)
    pred_scaled = model.predict(input_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    
    return jsonify({'prediksi': float(pred[0][0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
