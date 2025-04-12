from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os

from utils import get_db_connection, load_scaler

predict_bp = Blueprint('predict', __name__)
model_path = 'model_rnn_konsumsi.keras'
scaler_path = 'scaler.pkl'

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        jumlah_bulan = int(data.get('jumlah_bulan', 1))

        if not customer_id:
            return jsonify({'error': 'customer_id tidak disediakan'}), 400

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({'error': 'Model atau scaler belum tersedia. Silakan latih model terlebih dahulu melalui /train_model'}), 400

        model = tf.keras.models.load_model(model_path)
        scaler = load_scaler()

        conn = get_db_connection()
        query = f"""
            SELECT pemakaian_kwh FROM consumptions
            WHERE customer_id = {customer_id}
            ORDER BY tahun, bulan
            LIMIT 12
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) < 12:
            return jsonify({'error': 'Data tidak cukup untuk prediksi'}), 400

        usage = df['pemakaian_kwh'].values
        input_window = usage[-12:].tolist()
        input_window = scaler.transform(np.array(input_window).reshape(-1, 1)).reshape(1, 12, 1)

        prediksi = []
        for _ in range(jumlah_bulan):
            pred = model.predict(input_window)[0][0]
            prediksi.append(pred)
            new_input = np.append(input_window[0, 1:, 0], pred)
            input_window = new_input.reshape(1, 12, 1)

        prediksi_asli = scaler.inverse_transform(np.array(prediksi).reshape(-1, 1)).flatten()
        return jsonify({'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
