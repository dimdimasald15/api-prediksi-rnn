#predict.py:
from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from utils import get_db_connection, load_scaler, kategori_tarif_daya, one_hot_encode, scale_daya, TARIF_LIST, KATEGORI_LIST

predict_bp = Blueprint('predict', __name__)
model_path = 'model_rnn_konsumsi.keras'

# Load model dan scaler
model = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
scaler = load_scaler()

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        jumlah_bulan = int(data.get('jumlah_bulan', 1))

        if not customer_id:
            return jsonify({'error': 'customer_id tidak disediakan'}), 400

        conn = get_db_connection()
        query = f"""
            SELECT cs.tarif, cs.daya, cs.kategori, cn.pemakaian_kwh FROM consumptions cn
            JOIN customers cs ON cs.id = cn.customer_id
            WHERE customer_id = {customer_id}
            ORDER BY tahun, bulan
            LIMIT 12
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) < 12:
            return jsonify({'error': 'Data tidak cukup untuk prediksi'}), 400

        usage = df['pemakaian_kwh'].values.tolist()
        tarif = df['tarif'].iloc[0]
        daya = df['daya'].iloc[0]
        kategori = kategori_tarif_daya(tarif, daya)

        tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
        daya_scaled = [scale_daya(daya)]
        kategori_encoded = one_hot_encode(kategori, KATEGORI_LIST)
        fitur_statis = tarif_encoded + daya_scaled + kategori_encoded

        input_window = [[*fitur_statis, val] for val in usage[-12:]]
        input_window = scaler.transform(np.array(input_window)).reshape(1, 12, -1)

        prediksi = []
        for _ in range(jumlah_bulan):
            pred = model.predict(input_window, verbose=0)[0][0]
            prediksi.append(pred)

            new_input = input_window[0, 1:, :].tolist()
            new_input.append(fitur_statis + [pred])
            input_window = scaler.transform(np.array(new_input)).reshape(1, 12, -1)

        prediksi_asli = scaler.inverse_transform(np.array([fitur_statis + [p] for p in prediksi]))[:, -1]
        return jsonify({'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
