from flask import Blueprint
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, RepeatVector
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

        # Step 2: Buat kategori final gabungan
        df['kategori_final'] = df.apply(lambda row: kategori_tarif_daya(row['tarif'], row['daya']), axis=1)

        # Step 3: Preprocessing time series + fitur statis
        grouped = df.groupby('customer_id')
        X, y = [], []

        for _, group in grouped:
            usage = group['pemakaian_kwh'].values
            if len(usage) < 13:
                continue

            tarif = group['tarif'].iloc[0]
            daya = group['daya'].iloc[0]
            kategori = group['kategori_final'].iloc[0]

            tarif_encoded = one_hot_encode(tarif, TARIF_LIST)
            kategori_encoded = one_hot_encode(kategori, KATEGORI_LIST)
            daya_scaled = [scale_daya(daya)]

            fitur_statis = tarif_encoded + daya_scaled + kategori_encoded  # Total fitur: 8 + 1 + 3 = 12

            for i in range(len(usage) - 12):
                window = usage[i:i+12]
                target = usage[i+12]
                X.append([[*fitur_statis, val] for val in window])
                y.append(target)

        if not X or not y:
            raise Exception("Data tidak cukup untuk pelatihan model")

        update_progress(30)

        # Step 4: Scaling
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

        # Step 5: Definisikan model LSTM
        # Input sequence: 12 bulan terakhir konsumsi
        input_seq = Input(shape=(12, 1), name="input_seq")

        # Fitur statis: tarif, daya, kategori (total 12 dimensi mungkin)
        input_static = Input(shape=(12,), name="input_static")

        # LSTM branch
        x = LSTM(64, activation='tanh')(input_seq)

        # Static input dipakai di setiap timestep (ulangi 12x)
        static_repeated = RepeatVector(1)(input_static)

        # Gabung LSTM dan fitur statis
        combined = Concatenate()([x, static_repeated[:, 0, :]])  # ambil hanya 1 langkah

        output = Dense(1, name="output")(combined)

        model = Model(inputs=[input_seq, input_static], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # Progress callback per epoch
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = 50 + int((epoch + 1) / self.params['epochs'] * 40)
                update_progress(progress)
                print(f"Training progress: {progress}%")

        # Step 6: Training
        history = model.fit(
            X_scaled,
            y_scaled,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[ProgressCallback()],
            verbose=1
        )

        # Step 7: Simpan hasil training ke file (jika diperlukan)
        try:
            with open('static/plots/training_history.txt', 'w') as f:
                f.write("epoch,loss,val_loss\n")
                for i, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                    f.write(f"{i},{loss},{val_loss}\n")
        except Exception as e:
            print(f"Gagal menyimpan history: {e}")

        update_progress(95)

        # Step 8: Simpan model dan scaler
        model.save('model_rnn_konsumsi.keras')
        save_scaler(x_scaler)
        save_scaler(y_scaler, 'scaler_y.pkl')

        update_progress(100)
        return model

    except Exception as e:
        raise Exception(f"Error pada train_model: {str(e)}")
