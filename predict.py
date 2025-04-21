# predict.py
from flask import Blueprint, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib
# Set matplotlib to use non-interactive backend to avoid GUI errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import (
    get_db_connection, load_scaler, load_y_scaler
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
        customer_nama = data.get('nama')
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
        engine = get_db_connection()
        if not engine:
            raise Exception("Koneksi database gagal")
        
        query_usage = f"""
            SELECT bulan, tahun, pemakaian_kwh FROM consumptions
            WHERE customer_id = {customer_id}
            ORDER BY tahun, bulan DESC
            LIMIT 12
        """
        with engine.connect() as conn:
            df_usage = pd.read_sql(query_usage, conn)

        if df_usage.empty:
            raise Exception("Data pemakaian tidak ditemukan")
        
        query_customer = f"""
            SELECT tarif, daya, kategori FROM customers
            WHERE id = {customer_id}
        """
        with engine.connect() as conn:
            df_customer = pd.read_sql(query_customer, conn)
        
        if df_customer.empty:
            return jsonify({'error': f'Customer dengan ID {customer_id} tidak ditemukan'}), 404
        if len(df_usage) < 12:
            return jsonify({'error': f'Data historis tidak cukup. Dibutuhkan 12 bulan, hanya tersedia {len(df_usage)}'}), 400

        # Siapkan input
        usage = df_usage['pemakaian_kwh'].values[::-1]
        bulan = df_usage['bulan'].values[::-1]
        tahun = df_usage['tahun'].values[::-1]

        input_sequence = [[val] for val in usage[-12:]]
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
                np.array([[pred_scaled]])
            ])
            current_input = new_features.reshape(1, 12, len(input_sequence[0]))

        # Inverse transform hasil prediksi Y
        y_pred_array = np.array(prediksi).reshape(-1, 1)
        prediksi_asli = y_scaler.inverse_transform(y_pred_array).flatten()
        
        # Buat folder jika belum ada
        plot_folder = 'static/plots'
        os.makedirs(plot_folder, exist_ok=True)

        # Generate plot filename
        plot_filename = f'prediksi_CustomerId {customer_id}_{jumlah_bulan} bulan_ke_depan.png'
        plot_path = os.path.join(plot_folder, plot_filename)

        # Prepare x-axis labels for historical data
        x_labels_hist = [f"{b}/{t}" for b, t in zip(bulan[-12:], tahun[-12:])]

        # Generate prediction months/years
        pred_months = []
        next_month = bulan[-1]
        next_year = tahun[-1]

        for i in range(jumlah_bulan):
            next_month += 1
            if next_month > 12:
                next_month = 1
                next_year += 1
            pred_months.append(f"{next_month}/{next_year}")

        # Combine historical and prediction labels
        all_labels = x_labels_hist + pred_months

        # Create a new figure with explicit figure instance to avoid memory leaks
        fig = plt.figure(figsize=(10, 5))

        try:
            # Plot historical data
            plt.plot(range(12), usage[-12:], label='Historis', marker='o')
            
            # Plot prediction data
            pred_line = plt.plot(range(12, 12 + jumlah_bulan), prediksi_asli, label='Prediksi', 
                                marker='o', linestyle='--')
            
            # Get the color of the prediction line
            pred_color = pred_line[0].get_color()
            
            # Plot the connecting line between the last historical point and first prediction point
            plt.plot([11, 12], [usage[-1], prediksi_asli[0]], linestyle='--', color=pred_color, marker='')
            
            # Set x-ticks and labels
            plt.xticks(range(12 + jumlah_bulan), all_labels, rotation=45)
            
            # Color the prediction labels to match the prediction line
            ax = plt.gca()
            for i in range(jumlah_bulan):
                ax.get_xticklabels()[i + 12].set_color(pred_color)
            
            plt.xlabel('Bulan/Tahun')
            plt.ylabel('Pemakaian kWh')
            plt.title(f'Prediksi Pemakaian Listrik {customer_nama.upper()} \nDalam {jumlah_bulan} Bulan Ke Depan')
            plt.legend()
            plt.grid(True)
            
            # Add padding to prevent label cutoff
            plt.tight_layout()
            
            # Simpan file gambar prediksi
            fig.savefig(plot_path)
        except Exception as plot_error:
            # Log the specific plotting error
            print(f"Error plotting: {str(plot_error)}")
        finally:
            # Always close the figure to release memory
            plt.close(fig)

        return jsonify({
            'customer_id': customer_id,
            'customer_nama': customer_nama,
            'jumlah_bulan': jumlah_bulan,
            'plot_filename': plot_filename,
            'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli],
        })

    except Exception as e:
        return jsonify({'error': f'Error tidak terduga: {str(e)}'}), 500
    
@predict_bp.route('/plot/<filename>', methods=['GET'])
def get_plot(filename):
    return send_from_directory('static/plots', filename)
