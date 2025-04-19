from flask import Blueprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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

def detect_outliers(series, threshold=3):
    """Deteksi dan ganti outlier dengan nilai median."""
    mean = np.mean(series)
    std = np.std(series)
    outliers = np.abs(series - mean) > (threshold * std)
    series_filtered = series.copy()
    series_filtered[outliers] = np.median(series)
    return series_filtered

def add_seasonal_features(df):
    """Tambahkan fitur musiman berdasarkan bulan."""
    # Pastikan kolom bulan dan tahun tersedia
    if 'bulan' in df.columns and 'tahun' in df.columns:
        # Buat kolom tanggal
        df['tanggal'] = pd.to_datetime(df['tahun'].astype(str) + '-' + df['bulan'].astype(str) + '-01')
        # Ekstrak fitur musiman
        df['month_sin'] = np.sin(2 * np.pi * df['bulan']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['bulan']/12)
        # Tambahkan kolom untuk quarter/kuartal
        df['quarter'] = pd.DatetimeIndex(df['tanggal']).quarter
        # One-hot encode quarter
        quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter')
        df = pd.concat([df, quarter_dummies], axis=1)
    return df

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
        SELECT cn.customer_id, cs.tarif, cs.daya, cs.kategori, 
               cn.pemakaian_kwh, cn.bulan, cn.tahun
        FROM consumptions cn
        JOIN customers cs ON cs.id = cn.customer_id
        ORDER BY cn.customer_id, cn.tahun, cn.bulan
        """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        update_progress(10)

        if df.empty:
            raise Exception("Data tidak ditemukan")

        # Step 2: Feature Engineering
        # 2.1 Tambahkan fitur musiman
        df = add_seasonal_features(df)
        
        # 2.2 Buat kategori final gabungan
        df['kategori_final'] = df.apply(lambda row: kategori_tarif_daya(row['tarif'], row['daya']), axis=1)
        
        # 2.3 Deteksi dan tangani outlier per pelanggan
        df_clean = df.copy()
        for customer in df['customer_id'].unique():
            mask = df['customer_id'] == customer
            df_clean.loc[mask, 'pemakaian_kwh'] = detect_outliers(df.loc[mask, 'pemakaian_kwh'])
        
        df = df_clean
        update_progress(15)

        # Step 3: Preprocessing time series + fitur statis
        # 3.1 Persiapkan data dengan sequence length yang lebih fleksibel
        sequence_length = 12  # 12 bulan untuk prediksi
        forecast_horizon = 1   # Prediksi 1 bulan ke depan
        
        grouped = df.groupby('customer_id')
        
        X_sequences = []
        y_targets = []
        customer_ids = []
        
        for customer_id, group in grouped:
            # Sort by date to ensure proper time sequence
            group = group.sort_values(by=['tahun', 'bulan'])
            
            # Get static features
            tarif = group['tarif'].iloc[0]
            daya = group['daya'].iloc[0]
            kategori = group['kategori_final'].iloc[0]
            
            # Encode static features
            tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
            kategori_encoded = one_hot_encode(kategori, KATEGORI_LIST)
            daya_scaled = [scale_daya(daya)]
            
            # Basic static features
            static_features = tarif_encoded + daya_scaled + kategori_encoded
            
            # Get time series data
            ts_data = group['pemakaian_kwh'].values
            
            # Skip if we don't have enough data
            if len(ts_data) < sequence_length + forecast_horizon:
                continue
                
            # Create sequences with seasonal features
            for i in range(len(ts_data) - sequence_length - forecast_horizon + 1):
                # Time window
                window = ts_data[i:i+sequence_length]
                
                # Get seasonal features for this window
                seasonal_features = []
                for j in range(i, i+sequence_length):
                    month_features = [
                        group['month_sin'].iloc[j],
                        group['month_cos'].iloc[j]
                    ]
                    # Add quarter one-hot encoding if available
                    if 'quarter_1' in group.columns:
                        quarter_features = [
                            group['quarter_1'].iloc[j],
                            group['quarter_2'].iloc[j],
                            group['quarter_3'].iloc[j],
                            group['quarter_4'].iloc[j]
                        ]
                        month_features.extend(quarter_features)
                    
                    seasonal_features.append(month_features)
                
                # Combine features for each timestep in the sequence
                sequence = []
                for t in range(sequence_length):
                    # For each timestep, combine:
                    # 1. Static features (customer characteristics)
                    # 2. Consumption value
                    # 3. Seasonal features
                    timestep_features = static_features + [window[t]] + seasonal_features[t]
                    sequence.append(timestep_features)
                
                # Target is the next month's consumption
                target = ts_data[i+sequence_length]
                
                X_sequences.append(sequence)
                y_targets.append(target)
                customer_ids.append(customer_id)
        
        update_progress(30)
        
        if not X_sequences:
            raise Exception("Data tidak cukup untuk pelatihan model")
        
        # Convert to numpy arrays
        X = np.array(X_sequences)
        y = np.array(y_targets).reshape(-1, 1)
        customer_ids = np.array(customer_ids)
        
        # Step 4: Normalize data
        # 4.1 Split data terlebih dahulu supaya tidak ada data leak
        X_train, X_val, y_train, y_val, customer_train, customer_val = train_test_split(
            X, y, customer_ids, test_size=0.2, random_state=42
        )
        
        # 4.2 Normalize features
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # Reshape untuk scaling
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_val_flat = X_val.reshape(-1, X_val.shape[-1])
        
        # Fit dan transform hanya pada training data
        X_train_scaled_flat = x_scaler.fit_transform(X_train_flat)
        X_val_scaled_flat = x_scaler.transform(X_val_flat)
        
        # Reshape kembali ke bentuk 3D
        X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
        X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
        
        # Scale target
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        update_progress(40)

        # Step 5: Definisikan model LSTM yang lebih kompleks
        input_shape = (sequence_length, X_train.shape[2])
        
        model = Sequential([
            # Bidirectional LSTM layers
            Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            Bidirectional(LSTM(32, activation='relu')),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(1)
        ])
        
        # Optimizer dengan learning rate yang lebih kecil
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Callbacks untuk training yang lebih baik
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Progress callback per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # Calculate progress based on early stopping max epochs
                max_epochs = 100  # Assuming maximum epochs we would ever run
                progress = 40 + int((epoch + 1) / max_epochs * 50)
                progress = min(90, progress)  # Cap at 90% until training complete
                update_progress(progress)
                print(f"Training progress: {progress}%")

        callbacks.append(ProgressCallback())
        
        # Step 6: Training
        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=100,  # Set higher, early stopping will prevent overfitting
            batch_size=32,
            validation_data=(X_val_scaled, y_val_scaled),
            callbacks=callbacks,
            verbose=1
        )

        update_progress(90)
        
        # Step 7: Evaluasi model
        # 7.1 Prediksi pada validation set
        y_val_pred_scaled = model.predict(X_val_scaled)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
        y_val_true = y_scaler.inverse_transform(y_val_scaled)
        
        # 7.2 Calculate error metrics
        mae = np.mean(np.abs(y_val_pred - y_val_true))
        mape = np.mean(np.abs((y_val_true - y_val_pred) / y_val_true)) * 100
        rmse = np.sqrt(np.mean((y_val_pred - y_val_true)**2))
        
        # 7.3 Save metrics
        metrics = {
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse)
        }
        
        # Save metrics to file
        with open('static/plots/model_metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        # 7.4 Save training history
        try:
            with open('static/plots/training_history.txt', 'w') as f:
                f.write("epoch,loss,val_loss,mae,val_mae\n")
                for i in range(len(history.history['loss'])):
                    f.write(f"{i},{history.history['loss'][i]},{history.history['val_loss'][i]},{history.history['mae'][i]},{history.history['val_mae'][i]}\n")
        except Exception as e:
            print(f"Gagal menyimpan history: {e}")
        
        # Step 8: Simpan model dan scaler
        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler)
        save_scaler(y_scaler, 'scaler_y.pkl')
        
        # Save feature importance info for reference
        feature_info = {
            'sequence_length': sequence_length,
            'static_features': len(static_features),
            'total_features': X_train.shape[2],
            'total_samples': len(X_train)
        }
        
        with open('static/plots/feature_info.txt', 'w') as f:
            for key, value in feature_info.items():
                f.write(f"{key}: {value}\n")

        update_progress(100)
        return model, metrics

    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")