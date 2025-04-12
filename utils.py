# utils.py
import mysql.connector
import pickle
import os

# Database config
db_config = {
    'user': 'root',
    'password': 'ThpPEjAeCkstBjBiUdtmqcqwYRGhbyKh',
    'host': 'switchback.proxy.rlwy.net',
    'port': 53354,
    'database': 'railway'
}

# Encoder categories
TARIF_LIST = ['B1', 'B2', 'P1', 'P3', 'R1', 'R2', 'R3', 'S2']
KATEGORI_LIST = ['subsidi', 'non-subsidi', 'pju']
DAYA_MIN = 2200
DAYA_MAX = 66000

def get_db_connection():
    return mysql.connector.connect(**db_config)

def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path='scaler.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file '{path}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_y_scaler(path='scaler_y.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Y Scaler file '{path}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
    with open(path, 'rb') as f:
        return pickle.load(f)

def kategori_tarif_daya(tarif, daya):
    try:
        daya = int(daya)
    except:
        return 'tidak diketahui'
    if tarif == 'R1' and daya <= 2200:
        return 'subsidi'
    elif tarif in ['P3', 'S2']:
        return 'pju'
    else:
        return 'non-subsidi'

def one_hot_encode(value, categories):
    return [1 if value == cat else 0 for cat in categories]

def scale_daya(daya):
    return (daya - DAYA_MIN) / (DAYA_MAX - DAYA_MIN) if DAYA_MIN <= daya <= DAYA_MAX else 0.5
