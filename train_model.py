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

@train_bp.route('/train_model', methods=['POST'])
def train_model():
    db_config = {
        'user': os.getenv('DB_USERNAME', 'root'),
        'password': os.getenv('DB_PASSWORD', 'ThpPEjAeCkstBjBiUdtmqcqwYRGhbyKh'),
        'host': os.getenv('DB_HOST', 'switchback.proxy.rlwy.net'),
        'port': int(os.getenv('DB_PORT', 53354)),
        'database': os.getenv('DB_DATABASE', 'railway')
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
        print(grouped)
        # Siapkan sequence (X, y)
        sequences = []
        for _, row in grouped.iterrows():
            konsumsi = row['pemakaian_kwh']
            if len(konsumsi) >= 2:
                for i in range(len(konsumsi) - 1):
                    X_seq = konsumsi[i:i+1]
                    y_seq = konsumsi[i+1]
                    sequences.append((X_seq, y_seq))

        X = np.array([x for x, y in sequences])
        y = np.array([y for x, y in sequences])

        # Reshape untuk LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Normalisasi
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y = scaler_y.fit_transform(y.reshape(-1, 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=300, verbose=0)

        model.save('model_rnn_konsumsi.keras')
        return jsonify({'message': 'Model retrained successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
