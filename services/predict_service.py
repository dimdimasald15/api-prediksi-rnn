# services/predict_service.py
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import (
    MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, PLOT_FOLDER,
    get_db_connection, load_scaler)

os.makedirs(PLOT_FOLDER, exist_ok=True)

def predict_customer(customer_id, nama, jumlah_bulan):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(X_SCALER_PATH) or not os.path.exists(Y_SCALER_PATH):
        raise FileNotFoundError("Model atau scaler tidak ditemukan")

    model = tf.keras.models.load_model(MODEL_PATH)
    x_scaler = load_scaler(X_SCALER_PATH)
    y_scaler = load_scaler(Y_SCALER_PATH)

    engine = get_db_connection()
    query_usage = f"""
        SELECT bulan, tahun, pemakaian_kwh FROM consumptions
        WHERE customer_id = {customer_id}
        ORDER BY tahun DESC, bulan DESC
        LIMIT 12
    """
    with engine.connect() as conn:
        df_usage = pd.read_sql(query_usage, conn)

    if df_usage.empty or len(df_usage) < 12:
        raise ValueError("Data historis tidak mencukupi")

    query_customer = f"""
        SELECT tarif, daya, kategori FROM customers
        WHERE id = {customer_id}
    """
    with engine.connect() as conn:
        df_customer = pd.read_sql(query_customer, conn)

    if df_customer.empty:
        raise ValueError("Customer tidak ditemukan")

    usage = df_usage['pemakaian_kwh'].values[::-1]
    bulan = df_usage['bulan'].values[::-1]
    tahun = df_usage['tahun'].values[::-1]

    input_sequence = [[val] for val in usage[-12:]]
    input_array = np.array(input_sequence).reshape(1, 12, 1)
    input_scaled = x_scaler.transform(input_array.reshape(-1, 1)).reshape(input_array.shape)

    prediksi = []
    current_input = input_scaled.copy()

    for _ in range(jumlah_bulan):
        pred_scaled = model.predict(current_input, verbose=0)[0][0]
        prediksi.append(pred_scaled)

        new_features = np.concatenate([current_input[0, 1:, :], np.array([[pred_scaled]])])
        current_input = new_features.reshape(1, 12, 1)

    y_pred_array = np.array(prediksi).reshape(-1, 1)
    prediksi_asli = y_scaler.inverse_transform(y_pred_array).flatten()
    # gambar plot
    plot_filename = generate_plot(
        customer_id=customer_id,
        nama=nama,
        usage=usage,
        bulan=bulan,
        tahun=tahun,
        prediksi_asli=prediksi_asli,
        jumlah_bulan=jumlah_bulan
    )

    return {
        'customer_id': customer_id,
        'customer_nama': nama,
        'jumlah_bulan': jumlah_bulan,
        'prediksi_kwh': [round(float(p), 2) for p in prediksi_asli],
        'plot_filename': plot_filename
    }

def generate_plot(customer_id, nama, usage, bulan, tahun, prediksi_asli, jumlah_bulan):
    plot_filename = f'prediksi_CustomerId_{customer_id}_{jumlah_bulan}_bulan.png'
    plot_path = os.path.join(PLOT_FOLDER, plot_filename)

    # Generate label untuk X axis
    x_labels_hist = [f"{b}/{t}" for b, t in zip(bulan[-12:], tahun[-12:])]
    pred_months = []
    next_month = bulan[-1]
    next_year = tahun[-1]

    for _ in range(jumlah_bulan):
        next_month += 1
        if next_month > 12:
            next_month = 1
            next_year += 1
        pred_months.append(f"{next_month}/{next_year}")

    all_labels = x_labels_hist + pred_months

    fig = plt.figure(figsize=(10, 5))
    try:
        plt.plot(range(12), usage[-12:], label='Historis', marker='o')
        pred_line = plt.plot(range(12, 12 + jumlah_bulan), prediksi_asli, label='Prediksi', 
                    marker='o', linestyle='--')
        pred_color = pred_line[0].get_color()
        plt.plot([11, 12], [usage[-1], prediksi_asli[0]], linestyle='--', color=pred_color)
        plt.xticks(range(12 + jumlah_bulan), all_labels, rotation=45)

        ax = plt.gca()
        for i in range(jumlah_bulan):
            ax.get_xticklabels()[i + 12].set_color(pred_color)

        plt.xlabel('Bulan/Tahun')
        plt.ylabel('Pemakaian kWh')
        plt.title(f'Prediksi Pemakaian Listrik {nama.upper()} \nDalam {jumlah_bulan} Bulan Ke Depan')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(plot_path)
    finally:
        plt.close(fig)

    return plot_filename

