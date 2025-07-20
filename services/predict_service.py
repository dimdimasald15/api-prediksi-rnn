# services/predict_service.py
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pandas as pd
from pandas import isna
import os
import logging
import json

from utils import (
    MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH,DAYA_SCALER_PATH, ENCODER_INFO_PATH,
    get_db_connection, load_scaler, load_encoder_info)
from helper.redis_helper import redis_client, redis_connection_required
from helper.generate_plot_helper import generate_plot

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Nonaktifkan GPU

def load_model():
    # Tambahkan config untuk CPU saja
    tf.config.set_visible_devices([], 'GPU')
    return tf.keras.models.load_model(MODEL_PATH)

def predict_customer(customer_id, jumlah_bulan, prefetched_data=None):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(X_SCALER_PATH) or not os.path.exists(Y_SCALER_PATH):
        raise FileNotFoundError("Model atau scaler tidak ditemukan")

    model = load_model()
    x_scaler = load_scaler(X_SCALER_PATH)
    y_scaler = load_scaler(Y_SCALER_PATH)
    daya_scaler = load_scaler(DAYA_SCALER_PATH) # Muat scaler untuk daya
    encoder_info = load_encoder_info(ENCODER_INFO_PATH) # Muat informasi encoder

    tarif_cols = encoder_info['tarif_columns']
    kategori_cols = encoder_info['kategori_columns']
    static_feature_columns_order = encoder_info['static_feature_columns_order']
    num_static_features = len(static_feature_columns_order)

    df_customer_raw = None
    df_usage = None
    # Gunakan data hasil prefetch jika tersedia
    if prefetched_data is not None:
        df_customer_raw = prefetched_data.get('customer')
        df_usage = prefetched_data.get('consumptions')

        if df_customer_raw is None or df_usage is None or df_usage.empty or len(df_usage) < 12:
            raise ValueError(f"Data tidak lengkap untuk customer_id {customer_id} (prefetched)")
    else:
        # Fetch dari database jika data belum diprefetch
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
            SELECT u.name as nama, c.tarif, c.daya, c.kategori 
            FROM customers as c
            JOIN users u ON u.id = c.user_id
            WHERE c.id = {customer_id}
        """
        with engine.connect() as conn:
            df_customer_raw = pd.read_sql(query_customer, conn)

        if df_customer_raw.empty:
            raise ValueError("Customer tidak ditemukan")

    # --- Preprocessing Fitur Statis untuk Prediksi ---
    # Ambil nilai tarif, daya, kategori dari df_customer_raw
    customer_tarif = df_customer_raw.iloc[0]['tarif']
    customer_daya = df_customer_raw.iloc[0]['daya']
    customer_kategori = df_customer_raw.iloc[0]['kategori']

    # Buat DataFrame sementara untuk One-Hot Encoding agar konsisten dengan pelatihan
    # Pastikan semua kemungkinan kolom one-hot ada, bahkan jika nilainya 0
    temp_df_static = pd.DataFrame(columns=['tarif', 'kategori'])
    temp_df_static.loc[0] = [customer_tarif, customer_kategori]
    
    # Lakukan One-Hot Encoding
    temp_df_static = pd.get_dummies(temp_df_static, columns=['tarif', 'kategori'], prefix=['tarif', 'kategori'])

    # Tambahkan kolom one-hot yang mungkin tidak ada di temp_df_static
    # Ini penting agar urutan dan jumlah kolom one-hot konsisten dengan saat training
    for col in tarif_cols + kategori_cols:
        if col not in temp_df_static.columns:
            temp_df_static[col] = 0 # Tambahkan kolom dengan nilai 0 jika tidak ada

    # Scaling daya
    customer_daya_scaled = daya_scaler.transform([[customer_daya]])[0][0]
    temp_df_static['daya_scaled'] = customer_daya_scaled

    # Ambil fitur statis dalam urutan yang benar (sesuai saat training)
    # Pastikan semua kolom yang diharapkan ada di temp_df_static
    missing_static_cols = [col for col in static_feature_columns_order if col not in temp_df_static.columns]
    if missing_static_cols:
       for col in missing_static_cols:
            temp_df_static[col] = 0

    static_features_values = temp_df_static[static_feature_columns_order].iloc[0].values.tolist()
    
    nama = df_customer_raw.iloc[0]['nama']
    usage = df_usage['pemakaian_kwh'].values[::-1]
    
    input_sequence = []
    for val in usage[-12:]: # Ambil 12 bulan terakhir
        input_sequence.append([val] + static_features_values)
    
    input_array = np.array(input_sequence).reshape(1, 12, 1 + num_static_features)
    input_scaled_flat = x_scaler.transform(input_array.reshape(-1, input_array.shape[-1]))
    input_scaled = input_scaled_flat.reshape(input_array.shape)

    prediksi = []
    current_input = input_scaled.copy()

    for _ in range(jumlah_bulan):
        pred_scaled = model.predict(current_input, verbose=0)[0][0]
        prediksi.append(pred_scaled)

        new_features = np.concatenate([current_input[0, 1:, :], np.array([[pred_scaled] + static_features_values])])
        current_input = new_features.reshape(1, 12, 1 + num_static_features)

    y_pred_array = np.array(prediksi).reshape(-1, 1)
    prediksi_asli = y_scaler.inverse_transform(y_pred_array).flatten()

    bulan_hist = df_usage['bulan'].values[-12:][::-1] # Pastikan urutan kronologis
    tahun_hist = df_usage['tahun'].values[-12:][::-1] # Pastikan urutan kronologis

    plot_filename = generate_plot(
        customer_id=customer_id,
        nama=nama,
        usage=usage[-12:], # Hanya 12 bulan terakhir untuk plot
        bulan=bulan_hist,
        tahun=tahun_hist,
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

#services/predict_services.py
@redis_connection_required
def update_progress(batch_id, processed, total):
    """Update progress dengan atomic operation"""
    try:
        progress_data = {
            'processed': processed,
            'total': total,
            'percentage': int((processed / total) * 100) if total > 0 else 0
        }
        redis_client.setex(
            f"prediction:{batch_id}:progress",
            3600,  # 1 jam expiry
            json.dumps(progress_data)
        )
        return True
    except Exception as e:
        logging.error(f"Failed to update progress: {str(e)}")
        return False

# Prefetch data untuk multiple customers
def get_customers_data(customer_ids):
    engine = get_db_connection()
    placeholders = ', '.join(['%s'] * len(customer_ids))
    query = f"""
        SELECT 
            c.id as customer_id, 
            u.name as nama, 
            c.tarif, 
            c.daya, 
            c.kategori,
            JSON_ARRAYAGG(
                JSON_OBJECT(
                    'bulan', cons_sorted.bulan,
                    'tahun', cons_sorted.tahun,
                    'pemakaian_kwh', cons_sorted.pemakaian_kwh
                )
            ) AS consumptions
        FROM customers c
        LEFT JOIN (
            SELECT * FROM consumptions
            ORDER BY tahun DESC, bulan DESC
        ) AS cons_sorted ON cons_sorted.customer_id = c.id
        LEFT JOIN users u ON u.id = c.user_id
        WHERE c.id IN ({placeholders})
        GROUP BY c.id, u.name, c.tarif, c.daya, c.kategori
        ORDER BY c.id
    """
    
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=tuple(customer_ids))

def predict_batch_results(customerIds, batch_id, jumlah_bulan):
    if not customerIds:
        return {'error': 'Daftar customer_ids tidak boleh kosong', 'results': []}
    
    results = []
    total = len(customerIds)
    chunk_size = max(1, len(customerIds) // 10)  # Minimal 1 per chunk
    
    try:
        # Bagi customerIds menjadi chunk
        chunks = [customerIds[i:i + chunk_size] for i in range(0, total, chunk_size)]
        processed_count = 0
        for i, chunk in enumerate(chunks):            
            try:
                df = get_customers_data(chunk)
                
                for _, row in df.iterrows():
                    try:
                        consumptions_data = row['consumptions']
                        # Tambahkan ini untuk memastikan bentuknya:
                        print("DEBUG - customer:", row['customer_id'],  row['consumptions'])

                        if isinstance(consumptions_data, str):
                            try:
                                consumptions_data = json.loads(consumptions_data)
                            except json.JSONDecodeError:
                                continue

                        # Validasi apakah hasil parsing benar-benar list
                        if not isinstance(consumptions_data, list):
                            continue
                        
                        consumptions = pd.DataFrame(consumptions_data)
                        print("DEBUG - consumptions:", consumptions)

                        if consumptions.empty or len(consumptions) < 12:
                            raise ValueError("Data historis kurang dari 12 bulan")

                        consumptions.sort_values(['tahun', 'bulan'], ascending=[False, False], inplace=True)

                        prefetched_data = {
                            'customer': row.to_frame().T,
                            'consumptions': consumptions
                        }

                        result = predict_customer(
                            customer_id=row['customer_id'],
                            jumlah_bulan=jumlah_bulan,
                            prefetched_data=prefetched_data
                        )
                        results.append(result)

                    except Exception as e:
                        logging.warning(f"Prediction failed for customer_id {row.get('customer_id', 'UNKNOWN')}: {str(e)}")
                        continue
                    
                    processed_count += 1
                    update_progress(batch_id, processed_count, total)
                   
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {str(e)}")
                continue
                
        return {
            'success': True,
            'processed_count': len(results),
            'results': results
        }
        
    except Exception as e:
        logging.error(f"Batch prediction failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'results': results
        }
