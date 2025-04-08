# train_model.py
from flask import Blueprint, jsonify
import os
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

train_bp = Blueprint('train_model', __name__)

def train_and_save_model():
    db_config = {
        'user': 'root',
        'password': 'ThpPEjAeCkstBjBiUdtmqcqwYRGhbyKh',
        'host': 'switchback.proxy.rlwy.net',
        'port': 53354,
        'database': 'railway'
    }

    try:
        conn = mysql.connector.connect(**db_config)
        query = """
        SELECT cn.id,cn.customer_id,cs.id_pelanggan, cs.no_meter, cs.nama, cs.alamat, cn.bulan, cn.tahun, cn.pemakaian_kwh 
        FROM consumptions cn 
        JOIN customers cs ON cs.id = cn.customer_id;
        """
        df = pd.read_sql(query, conn)
        conn.close()

        df['periode'] = df['tahun'].astype(str) + '-' + df['bulan'].astype(str).str.zfill(2)

        # Urutkan dan kelompokkan
        grouped = df.groupby('customer_id')['pemakaian_kwh'].apply(list).reset_index()
        # Siapkan sequence (X, y)
        sequences = []
        for _, row in grouped.iterrows():
            usage = group['pemakaian_kwh'].values
            if len(usage) >= 13:
                for i in range(len(usage) - 12):
                    sequences.append(usage[i:i+12])
                    targets.append(usage[i+12])

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))

        # Buat dan latih model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(12, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=20, batch_size=16, verbose=0)

        # Simpan model
        model.save('model_rnn_konsumsi.keras')
        return jsonify({'message': 'Model retrained successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
