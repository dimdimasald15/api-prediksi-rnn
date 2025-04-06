from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
