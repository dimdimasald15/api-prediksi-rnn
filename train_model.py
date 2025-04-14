# train_model.py
from flask import Blueprint, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import get_db_connection, save_scaler, kategori_tarif_daya, one_hot_encode, scale_daya, TARIF_LIST, KATEGORI_LIST
import matplotlib.pyplot as plt


train_bp = Blueprint('train_model', __name__)

def train_and_save_model(progress_callback=None):
    def update_progress(persen):
        if progress_callback:
            progress_callback(persen)

    try:
        update_progress(0)

        engine = get_db_connection()
        if not engine:
            raise Exception("Koneksi database gagal")
        update_progress(5)

        query = """
        SELECT cn.customer_id, cs.tarif, cs.daya, cs.kategori, cn.pemakaian_kwh
        FROM consumptions cn
        JOIN customers cs ON cs.id = cn.customer_id
        ORDER BY cn.customer_id, cn.tahun, cn.bulan
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        update_progress(10)

        if df.empty:
            raise Exception("Data tidak ditemukan")

        df['kategori_final'] = df.apply(lambda row: kategori_tarif_daya(row['tarif'], row['daya']), axis=1)

        grouped = df.groupby('customer_id')
        X, y = [], []

        for _, group in grouped:
            usage = group['pemakaian_kwh'].values
            tarif = group['tarif'].iloc[0]
            daya = group['daya'].iloc[0]
            kategori = group['kategori_final'].iloc[0]

            tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
            kategori_encoded = one_hot_encode(kategori, KATEGORI_LIST)
            daya_scaled = [scale_daya(daya)]

            fitur_statis = tarif_encoded + daya_scaled + kategori_encoded  # Total = 8 + 1 + 3 = 12 fitur

            if len(usage) >= 13:
                for i in range(len(usage) - 12):
                    window = usage[i:i+12]
                    target = usage[i+12]
                    X.append([[*fitur_statis, val] for val in window])
                    y.append(target)

        if not X or not y:
            raise Exception("Data tidak cukup untuk pelatihan model")
        update_progress(40)

        X = np.array(X)
        y = np.array(y)

        # Gunakan scaler yang sama untuk X dan y
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        y = np.array(y).reshape(-1, 1)  # Data hasil windowing (sudah dikumpulkan sebelumnya)
        y_scaled = y_scaler.fit_transform(y)
        update_progress(50)

        # Reshape dengan aman
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = x_scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)
        
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(12, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Update progress per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                update_progress(50 + int((epoch + 1) / self.params['epochs'] * 40))

        history = model.fit(
            X_scaled, y_scaled, 
            epochs=20, 
            batch_size=16, 
            verbose=1,
            validation_split=0.2
        )

        # Plot loss
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)

        # Simpan gambar
        loss_plot_path = 'static/plots/loss_plot.png'
        plt.savefig(loss_plot_path)
        plt.close()
        
        update_progress(95)

        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler)  # Pastikan fungsi ini menyimpan scaler dengan benar
        save_scaler(y_scaler, 'scaler_y.pkl') 
        update_progress(100)

        return model  # Mengembalikan model jika berhasil

    except Exception as e:
        # Jangan return jsonify, tetapi raise exception agar ditangkap oleh app.py
        raise Exception(f"Error pada train_model: {str(e)}")
