from flask import Blueprint
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import get_db_connection, save_scaler, kategori_tarif_daya, one_hot_encode, scale_daya, TARIF_LIST, KATEGORI_LIST

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

        query = """ ... """
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        update_progress(10)

        if df.empty:
            raise Exception("Data tidak ditemukan")

        df['kategori_final'] = df.apply(lambda row: kategori_tarif_daya(row['tarif'], row['daya']), axis=1)

        X_raw, y_raw = [], []
        grouped = df.groupby('customer_id')
        for _, group in grouped:
            ...
        update_progress(40)

        X = np.array(X_raw)
        y = np.array(y_raw)

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = y_scaler.fit_transform(y)
        update_progress(50)

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(12, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Update progress per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                update_progress(50 + int((epoch + 1) / self.params['epochs'] * 40))

        model.fit(X_scaled, y_scaled, epochs=20, batch_size=16, verbose=1, callbacks=[ProgressCallback()])
        update_progress(95)

        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler, 'scaler.pkl')
        save_scaler(y_scaler, 'scaler_y.pkl')
        update_progress(100)

        return model
    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")
