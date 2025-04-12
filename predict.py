from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from utils import get_db_connection, load_scaler, kategori_tarif_daya, one_hot_encode, scale_daya, TARIF_LIST, KATEGORI_LIST

predict_bp = Blueprint('predict', __name__)
model_path = 'model_rnn_konsumsi.keras'
scaler_path = 'scaler.pkl'

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Validasi input
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

        # Validasi model dan scaler
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model tidak ditemukan di path: {model_path}. Silakan latih model terlebih dahulu.'}), 400
            
        if not os.path.exists(scaler_path):
            return jsonify({'error': f'Scaler tidak ditemukan di path: {scaler_path}. Silakan latih model terlebih dahulu.'}), 400

        # Load model dan scaler
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            return jsonify({'error': f'Gagal memuat model: {str(e)}'}), 500
            
        try:
            scaler = load_scaler()
        except Exception as e:
            return jsonify({'error': f'Gagal memuat scaler: {str(e)}'}), 500

        # Koneksi database dan ambil data
        try:
            conn = get_db_connection()
            if not conn:
                return jsonify({'error': 'Koneksi database gagal'}), 500
                
            # Ambil data historis pemakaian
            query_usage = f"""
                SELECT pemakaian_kwh FROM consumptions
                WHERE customer_id = {customer_id}
                ORDER BY tahun, bulan DESC
                LIMIT 12
            """
            df_usage = pd.read_sql(query_usage, conn)
            
            # Ambil data customer (tarif, daya, kategori)
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
                
        except Exception as e:
            return jsonify({'error': f'Error database: {str(e)}'}), 500

        # Persiapan data input
        usage = df_usage['pemakaian_kwh'].values[::-1]  # Reverse karena query DESC
        
        # Ambil data statis customer
        tarif = df_customer['tarif'].iloc[0]
        daya = df_customer['daya'].iloc[0]
        kategori_final = kategori_tarif_daya(df_customer['tarif'].iloc[0], df_customer['daya'].iloc[0])
        
        # Siapkan fitur statis
        tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
        kategori_encoded = one_hot_encode(kategori_final, KATEGORI_LIST)
        daya_scaled = [scale_daya(daya)]
        fitur_statis = tarif_encoded + daya_scaled + kategori_encoded
        
        # Siapkan input window dengan fitur statis
        input_sequence = []
        for val in usage[-12:]:
            input_sequence.append([*fitur_statis, val])
            
        # Transform input
        input_array = np.array(input_sequence).reshape(1, 12, len(input_sequence[0]))
        input_scaled = scaler.transform(input_array.reshape(-1, input_array.shape[-1])).reshape(input_array.shape)

        # Prediksi
        prediksi = []
        current_input = input_scaled.copy()
        
        for _ in range(jumlah_bulan):
            try:
                pred = model.predict(current_input, verbose=0)[0][0]
                prediksi.append(pred)
                
                # Update input untuk prediksi bulan berikutnya
                new_features = np.concatenate([
                    current_input[0, 1:, :],  # Ambil 11 timestep terakhir
                    np.array([[*fitur_statis, pred]])  # Tambahkan prediksi baru dengan fitur statis
                ])
                current_input = new_features.reshape(1, 12, len(input_sequence[0]))
                
            except Exception as e:
                return jsonify({'error': f'Error saat melakukan prediksi: {str(e)}'}), 500

        # Konversi kembali ke nilai asli
        try:
            # Buat array lengkap dengan semua fitur (statis+dinamis)
            pred_with_features = []
            for p in prediksi:
                pred_with_features.append([*fitur_statis, p])
                
            pred_array = np.array(pred_with_features)
            prediksi_asli = scaler.inverse_transform(pred_array)[:, -1]  # Ambil hanya nilai pemakaian
            
            return jsonify({
                'customer_id': customer_id,
                'jumlah_bulan': jumlah_bulan,
                'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli]
            })
            
        except Exception as e:
            return jsonify({'error': f'Error saat transformasi hasil: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Error tidak terduga: {str(e)}'}), 500