# predict.py
from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from utils import (
    get_db_connection, load_scaler, load_y_scaler,
    kategori_tarif_daya, one_hot_encode, scale_daya,
    TARIF_LIST, KATEGORI_LIST
)

predict_bp = Blueprint('predict', __name__)
model_path = 'model_rnn_konsumsi.keras'
x_scaler_path = 'scaler.pkl'
y_scaler_path = 'scaler_y.pkl'

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request harus dalam format JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Body request kosong'}), 400
            
        customer_id = data.get('customer_id')
        if not customer_id:
            return jsonify({'error': 'customer_id tidak disediakan'}), 400
            
        jumlah_bulan = data.get('jumlah_bulan', 1)
        if not isinstance(jumlah_bulan, int) or jumlah_bulan < 1:
            return jsonify({'error': 'jumlah_bulan harus berupa bilangan bulat positif'}), 400

        # Validasi file
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model tidak ditemukan di path: {model_path}'}), 400
        if not os.path.exists(x_scaler_path):
            return jsonify({'error': f'X Scaler tidak ditemukan di path: {x_scaler_path}'}), 400
        if not os.path.exists(y_scaler_path):
            return jsonify({'error': f'Y Scaler tidak ditemukan di path: {y_scaler_path}'}), 400

        # Load model dan scalers
        model = tf.keras.models.load_model(model_path)
        x_scaler = load_scaler(x_scaler_path)
        y_scaler = load_y_scaler(y_scaler_path)

        # Ambil data dari DB
        conn = get_db_connection()
        query_usage = f"""
            SELECT pemakaian_kwh FROM consumptions
            WHERE customer_id = {customer_id}
            ORDER BY tahun, bulan DESC
            LIMIT 12
        """
        df_usage = pd.read_sql(query_usage, conn)

        query_customer = f"""
            SELECT tarif, daya, kategori FROM customers
            WHERE id = {customer_id}
        """
        df_customer = pd.read_sql(query_customer, conn)
        conn.close()

        if df_customer.empty:
            return jsonify({'error': f'Customer dengan ID {customer_id} tidak ditemukan'}), 404
        if len(df_usage) < 12:
            return jsonify({'error': f'Data historis tidak cukup. Dibutuhkan 12 bulan, hanya tersedia {len(df_usage)}'}), 400

        # Siapkan input
        usage = df_usage['pemakaian_kwh'].values[::-1]
        tarif = df_customer['tarif'].iloc[0]
        daya = df_customer['daya'].iloc[0]
        kategori_final = kategori_tarif_daya(tarif, daya)

        tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
        kategori_encoded = one_hot_encode(kategori_final, KATEGORI_LIST)
        daya_scaled = [scale_daya(daya)]
        fitur_statis = tarif_encoded + daya_scaled + kategori_encoded

        input_sequence = [[*fitur_statis, val] for val in usage[-12:]]
        input_array = np.array(input_sequence).reshape(1, 12, len(input_sequence[0]))
        input_scaled = x_scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_array.shape)

        # Prediksi
        prediksi = []
        current_input = input_scaled.copy()

        for _ in range(jumlah_bulan):
            pred_scaled = model.predict(current_input, verbose=0)[0][0]
            prediksi.append(pred_scaled)

            new_features = np.concatenate([
                current_input[0, 1:, :],
                np.array([[*fitur_statis, pred_scaled]])
            ])
            current_input = new_features.reshape(1, 12, len(input_sequence[0]))

        # Inverse transform hasil prediksi Y
        y_pred_array = np.array(prediksi).reshape(-1, 1)
        prediksi_asli = y_scaler.inverse_transform(y_pred_array).flatten()

        return jsonify({
            'customer_id': customer_id,
            'jumlah_bulan': jumlah_bulan,
            'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli]
        })

    except Exception as e:
        return jsonify({'error': f'Error tidak terduga: {str(e)}'}), 500
