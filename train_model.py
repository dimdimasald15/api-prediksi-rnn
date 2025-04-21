from flask import Blueprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils import (
    get_db_connection,
    save_scaler
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

        # Step 2: Preprocessing time series + fitur statis
        grouped = df.groupby('customer_id')
        X, y = [], []

        for customer_id, group_data in grouped:
            usage = group_data['pemakaian_kwh'].values
            if len(usage) < 13:
                continue

            for i in range(len(usage) - 12):
                window = usage[i:i+12]
                target = usage[i+12]
                X.append([[val] for val in window])
                y.append(target)

        if not X or not y:
            raise Exception("Data tidak cukup untuk pelatihan model")

        update_progress(30)

        # Step 3: Scaling
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        # Scaling X
        X_flat = X.reshape(-1, X.shape[-1])  # (total * 12, 13)
        X_scaled_flat = x_scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)

        # Scaling y
        y_scaled = y_scaler.fit_transform(y)

        update_progress(50)

        # Step 4: Definisikan model LSTM
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(12, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Progress callback per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = 50 + int((epoch + 1) / self.params['epochs'] * 40)
                update_progress(progress)
                print(f"Training progress: {progress}%")

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

        # Step 6: Simpan hasil training ke file (jika diperlukan)
        try:
            with open('static/plots/training_history.txt', 'w') as f:
                f.write("epoch,loss,val_loss\n")
                for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                    f.write(f"{i},{loss},{val_loss}\n")
        except Exception as e:
            print(f"Gagal menyimpan history: {e}")

        update_progress(95)

        # Step 7: Simpan model dan scaler
        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler)
        save_scaler(y_scaler, 'scaler_y.pkl')

        update_progress(100)
        return model

    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")
