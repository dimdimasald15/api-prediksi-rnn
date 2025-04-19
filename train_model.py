from flask import Blueprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from utils import (
    get_db_connection,
    save_scaler,
    kategori_tarif_daya,
    one_hot_encode,
    scale_daya,
    TARIF_LIST,
    KATEGORI_LIST
)

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

        # Step 1: Load dan gabungkan data konsumsi dan data pelanggan
        query = """
        SELECT cn.customer_id, cs.tarif, cs.daya, cs.kategori, cn.pemakaian_kwh,
               cn.bulan, cn.tahun
        FROM consumptions cn
        JOIN customers cs ON cs.id = cn.customer_id
        ORDER BY cn.customer_id, cn.tahun, cn.bulan
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        update_progress(10)

        if df.empty:
            raise Exception("Data tidak ditemukan")

        # Step 2: Buat kategori final gabungan
        df['kategori_final'] = df.apply(lambda row: kategori_tarif_daya(row['tarif'], row['daya']), axis=1)
        
        # Step 3: Tambahkan fitur musiman sederhana (tanpa library tambahan)
        # Sin dan Cos untuk bulan (menangkap siklus tahunan)
        if 'bulan' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['bulan']/12)
            df['month_cos'] = np.cos(2 * np.pi * df['bulan']/12)

        update_progress(15)

        # Step 4: Preprocessing time series + fitur statis
        grouped = df.groupby('customer_id')
        X, y = [], []

        sequence_length = 12  # Gunakan 12 bulan untuk prediksi

        for _, group in grouped:
            # Urutkan berdasarkan waktu
            group = group.sort_values(by=['tahun', 'bulan'])
            
            usage = group['pemakaian_kwh'].values
            if len(usage) < sequence_length + 1:  # Minimal 13 bulan data
                continue

            # Fitur statis
            tarif = group['tarif'].iloc[0]
            daya = group['daya'].iloc[0]
            kategori = group['kategori_final'].iloc[0]

            tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
            kategori_encoded = one_hot_encode(kategori, KATEGORI_LIST)
            daya_scaled = [scale_daya(daya)]

            # Gabungkan fitur statis: tarif + daya + kategori
            fitur_statis = tarif_encoded + daya_scaled + kategori_encoded

            # Handling outlier sederhana (ganti dengan median jika > 3 std)
            median = np.median(usage)
            std = np.std(usage)
            usage_cleaned = np.where(np.abs(usage - np.mean(usage)) > 3*std, median, usage)

            # Buat sequence dengan fitur musiman
            for i in range(len(usage_cleaned) - sequence_length):
                # Data time series
                window = usage_cleaned[i:i+sequence_length]
                target = usage_cleaned[i+sequence_length]
                
                # Gabungkan dengan fitur musiman
                sequence = []
                for j in range(sequence_length):
                    month_idx = (i+j) % len(group)  # Pastikan tidak out of bounds
                    
                    # Fitur musiman jika tersedia
                    seasonal_features = []
                    if 'month_sin' in group.columns and month_idx < len(group):
                        seasonal_features = [
                            group['month_sin'].iloc[month_idx],
                            group['month_cos'].iloc[month_idx]
                        ]
                    
                    # Gabungkan: [fitur_statis, konsumsi, fitur_musiman]
                    timestep_features = fitur_statis + [window[j]] + seasonal_features
                    sequence.append(timestep_features)
                
                X.append(sequence)
                y.append(target)

        if not X or not y:
            raise Exception("Data tidak cukup untuk pelatihan model")

        update_progress(30)

        # Step 5: Scaling
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # Hitung dimensi input
        feature_dim = X.shape[2]

        # Split data training (80%) dan validasi (20%)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Scaling untuk X
        x_scaler = MinMaxScaler()
        X_train_flat = X_train.reshape(-1, feature_dim)
        X_val_flat = X_val.reshape(-1, feature_dim)
        
        X_train_scaled_flat = x_scaler.fit_transform(X_train_flat)
        X_val_scaled_flat = x_scaler.transform(X_val_flat)
        
        X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
        X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)

        # Scaling untuk y
        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)

        update_progress(40)

        # Step 6: Definisikan model LSTM yang lebih baik tapi tidak terlalu berat
        model = Sequential([
            # First LSTM layer
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=(sequence_length, feature_dim)),
            Dropout(0.2),  # Add dropout to prevent overfitting
            
            # Second LSTM layer
            LSTM(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')

        # Step 7: Siapkan callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Progress callback
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                max_epochs = 50  # Asumsi maksimum epoch
                progress = 40 + int((epoch + 1) / max_epochs * 50)
                progress = min(90, progress)
                update_progress(progress)
                print(f"Training progress: {progress}%")
        
        callback_list = [early_stopping, ProgressCallback()]

        # Step 8: Training
        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_scaled, y_val_scaled),
            callbacks=callback_list,
            verbose=1
        )

        # Step 9: Evaluasi model
        val_loss = model.evaluate(X_val_scaled, y_val_scaled, verbose=0)
        
        # Prediksi pada validation set untuk perhitungan metrik
        y_val_pred_scaled = model.predict(X_val_scaled)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
        y_val_true = y_scaler.inverse_transform(y_val_scaled)
        
        # Hitung RMSE dan MAE
        rmse = np.sqrt(np.mean((y_val_pred - y_val_true)**2))
        mae = np.mean(np.abs(y_val_pred - y_val_true))
        
        # Simpan metrik
        with open('static/plots/model_metrics.txt', 'w') as f:
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"Validation Loss: {val_loss}\n")

        update_progress(95)

        # Step 10: Simpan training history
        try:
            with open('static/plots/training_history.txt', 'w') as f:
                f.write("epoch,loss,val_loss\n")
                for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                    f.write(f"{i},{loss},{val_loss}\n")
        except Exception as e:
            print(f"Gagal menyimpan history: {e}")

        # Step 11: Simpan model dan scaler
        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler)
        save_scaler(y_scaler, 'scaler_y.pkl')

        update_progress(100)
        return model

    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")