# train_model.py
from flask import Blueprint, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from utils import get_db_connection, save_scaler, kategori_tarif_daya, one_hot_encode, scale_daya, TARIF_LIST, KATEGORI_LIST

train_bp = Blueprint('train_model', __name__)

def train_and_save_model():
    try:
        conn = get_db_connection()
        query = """
        SELECT cs.customer_id, cs.tarif, cs.daya, cs.kategori, cn.pemakaian_kwh
        FROM consumptions cn
        JOIN customers cs ON cs.id = cn.customer_id
        ORDER BY cs.customer_id, cn.tahun, cn.bulan
        """
        df = pd.read_sql(query, conn)
        conn.close()

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

        X = np.array(X)
        y = np.array(y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(12, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=20, batch_size=16, verbose=1)

        model.save('model_rnn_konsumsi.keras')
        save_scaler(scaler)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
