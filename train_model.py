from flask import Blueprint
import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from utils import (
    get_db_connection,
    save_scaler,
    save_encoder_info
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg') # Ini harus sebelum import pyplot
import matplotlib.pyplot as plt
import os
import json

train_bp = Blueprint('train_model', __name__)

def train_and_save_model(progress_callback=None):
    def update_progress(persen, epoch=None, total_epochs=None, loss=None, val_loss=None):
        if progress_callback:
            progress_callback(persen, epoch, total_epochs, loss, val_loss)

    try:
        update_progress(0)

        engine = get_db_connection()
        if not engine:
            raise Exception("Koneksi database gagal")
        update_progress(5)

        # Step 1: Load dan gabungkan data konsumsi dan data pelanggan
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

        #  Step 2: Preprocessing fitur statis (tarif, daya, kategori)
        # Buat salinan DataFrame untuk preprocessing fitur statis
        df_processed = df.copy()

        # One-Hot Encode 'tarif' dan 'kategori'
        # pd.get_dummies akan membuat kolom baru untuk setiap kategori unik
        df_processed = pd.get_dummies(df_processed, columns=['tarif', 'kategori'], prefix=['tarif', 'kategori'])

        # Simpan nama kolom one-hot yang dihasilkan untuk konsistensi saat prediksi
        tarif_cols = [col for col in df_processed.columns if col.startswith('tarif_')]
        kategori_cols = [col for col in df_processed.columns if col.startswith('kategori_')]

        # Scaling 'daya'
        daya_scaler = MinMaxScaler()
        # Fit dan transform hanya kolom 'daya'
        df_processed['daya_scaled'] = daya_scaler.fit_transform(df_processed[['daya']])
        save_scaler(daya_scaler, 'scaler_daya.pkl') # Simpan scaler daya

        # Identifikasi semua kolom fitur statis yang akan digunakan
        # Urutkan untuk memastikan konsistensi urutan fitur saat input ke model
        static_feature_cols = sorted(tarif_cols + kategori_cols + ['daya_scaled'])

        # Simpan informasi encoder (nama kolom one-hot dan urutan fitur statis)
        encoder_info = {
            'tarif_columns': tarif_cols,
            'kategori_columns': kategori_cols,
            'static_feature_columns_order': static_feature_cols # Urutan fitur untuk input model
        }
        save_encoder_info(encoder_info, 'encoder_info.pkl')

        # Sekarang, siapkan data untuk SimpleRNN
        grouped = df_processed.groupby('customer_id') # Group DataFrame yang sudah diproses
        X, y = [], []
        num_static_features = len(static_feature_cols) # Jumlah fitur statis per time step

        for customer_id, group_data in grouped:
            usage = group_data['pemakaian_kwh'].values
            
            # Ambil fitur statis untuk customer ini (mereka konstan dalam grup)
            # Pastikan urutan kolom sesuai dengan static_feature_cols yang sudah diurutkan
            static_features_values = group_data[static_feature_cols].iloc[0].values 
            
            # Memastikan ada cukup data untuk jendela 12 bulan dan 1 target
            if len(usage) < 13: 
                continue

            for i in range(len(usage) - 12):
                window_kwh = usage[i:i+12] # Jendela 12 bulan pemakaian KWH
                target_kwh = usage[i+12]   # Target bulan ke-13
                
                # Gabungkan window KWH dengan fitur statis untuk setiap langkah waktu dalam window
                combined_window = []
                for kwh_val in window_kwh:
                    # Setiap elemen dalam sequence akan menjadi [kwh_val, static_feature_1, static_feature_2, ...]
                    combined_window.append([kwh_val] + static_features_values.tolist())
                    
                X.append(combined_window)
                y.append(target_kwh)

        if not X or not y:
            raise Exception("Data tidak cukup untuk pelatihan model setelah preprocessing fitur statis.")

        update_progress(30)

        # Step 3: Scaling seluruh fitur input (KWH + statis) dan target
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        x_scaler = MinMaxScaler() # Scaler untuk seluruh fitur input (KWH + statis)
        y_scaler = MinMaxScaler() # Scaler untuk target KWH

        # Scaling X: Flatten X, scale, lalu reshape kembali
        # X_flat akan memiliki dimensi (jumlah_sampel * sequence_length, jumlah_fitur_per_timestep)
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = x_scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)

        # Scaling y
        y_scaled = y_scaler.fit_transform(y)

        update_progress(50)

        # Step 4: Definisikan model SimpleRNN
        # input_shape akan menjadi (sequence_length, 1 (kwh) + jumlah_fitur_statis)
        model = Sequential([
            SimpleRNN(64, activation='relu', input_shape=(12, 1 + num_static_features)),
            Dropout(0.2), # Tambahkan Dropout untuk mencegah overfitting
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Progress callback per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_on_training = 50 + int((epoch + 1) / self.params['epochs'] * 40)
                update_progress(
                    progress_on_training,
                    epoch=epoch + 1, # epoch dimulai dari 0, tambahkan 1 untuk display
                    total_epochs=self.params['epochs'],
                    loss=logs.get('loss'),
                    val_loss=logs.get('val_loss')
                )
                print(f"Epoch {epoch+1}/{self.params['epochs']} - Loss: {logs.get('loss'):.4f}, Val Loss: {logs.get('val_loss'):.4f}")

        # Step 5: Training
        history = model.fit(
            X_scaled,
            y_scaled,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[ProgressCallback()],
            verbose=1
        )
        
        update_progress(90) 
        # Step 6: Evaluasi model
        print("\n--- Mengevaluasi Model ---")
        # 1. Melakukan Prediksi pada Data Skala
        y_pred_scaled = model.predict(X_scaled)

        # 2. Mengembalikan Prediksi dan Nilai Sebenarnya ke Skala Asli
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_scaler.inverse_transform(y_scaled)

        # 3. Menghitung Metrik
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse) # RMSE adalah akar kuadrat dari MSE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        evaluation_metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape)
        }

        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}")
        
        # Pastikan direktori untuk menyimpan plot ada
        plots_dir = 'static/plots/train_model'
        os.makedirs(plots_dir, exist_ok=True)

        # 4. Menyimpan Metrik ke File JSON
        metrics_file_path = os.path.join(plots_dir, 'model_metrics.json')
        with open(metrics_file_path, 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)
        print(f"Metrik disimpan ke: {metrics_file_path}")

        # 5. Visualisasi Prediksi vs Aktual (dan menyimpan sebagai gambar)
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual Values', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--', alpha=0.7)
        plt.title('Actual vs Predicted Consumption (Training Data)')
        plt.xlabel('Data Point Index')
        plt.ylabel('Pemakaian KWH')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
        plt.savefig(plot_path)
        plt.close() # Penting untuk menutup plot agar tidak memakan memori
        print(f"Plot Actual vs Predicted disimpan ke: {plot_path}")
        
        # Step 6: Simpan hasil training ke file (jika diperlukan)
        try:
            with open('static/plots/training_history.csv', 'w') as f:
                f.write("epoch,loss,val_loss\n")
                for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                    f.write(f"{i},{loss},{val_loss}\n")
        except Exception as e:
            print(f"Gagal menyimpan history: {e}")

        update_progress(95)

        # Step 7: Simpan model dan scaler
        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler, 'scaler_X.pkl') # Simpan scaler untuk input X
        save_scaler(y_scaler, 'scaler_y.pkl') # Simpan scaler untuk target y

        update_progress(100)
        return model

    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")
    
def get_training_history_data():
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), # Dapatkan path direktori saat ini (tempat app.py berada)
        'static', 'plots', 'training_history.csv'
    )
    
    if not os.path.exists(csv_path):
        return None, "File training_history.csv tidak ditemukan.", 404
    
    try:
        # Baca CSV menggunakan pandas
        df = pd.read_csv(csv_path)
        # Konversi DataFrame ke format list of dictionaries untuk JSON response
        data = df.to_dict(orient='records')
        return data, None, 200
    except Exception as e:
        return None, f"Gagal membaca data training: {str(e)}", 500
