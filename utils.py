import pickle
import os
from sqlalchemy import create_engine
from functools import lru_cache

# Base directory untuk menyimpan model, scaler, dan encoder info
# Ini akan mendapatkan path absolut dari direktori tempat file utils.py ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global constants, sekarang menggunakan BASE_DIR untuk path yang konsisten
MODEL_PATH = os.path.join(BASE_DIR, 'model_rnn_konsumsi.keras')
X_SCALER_PATH = os.path.join(BASE_DIR, 'scalers', 'scaler_X.pkl') # Perbaiki nama file scaler_X.pkl
Y_SCALER_PATH = os.path.join(BASE_DIR, 'scalers', 'scaler_y.pkl')
DAYA_SCALER_PATH = os.path.join(BASE_DIR, 'scalers', 'scaler_daya.pkl')
ENCODER_INFO_PATH = os.path.join(BASE_DIR, 'encoders', 'encoder_info.pkl')
PLOT_FOLDER = os.path.join(BASE_DIR, 'static', 'plots') # Pastikan ini juga menggunakan BASE_DIR

# Database config
db_config = {
    'user': 'root',
    'password': '', # Gunakan password kosong jika itu konfigurasi Anda
    'host': 'localhost',
    'port': 3306,
    'database': 'dbprediction' # Pastikan nama database ini benar
}

def get_db_connection():
    return create_engine(
        f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}",
        connect_args={"connect_timeout": 20},
        pool_pre_ping=True
    )

def save_scaler(scaler, filename): # Ubah parameter path menjadi filename
    """Menyimpan objek MinMaxScaler ke file pickle."""
    scaler_dir = os.path.join(BASE_DIR, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True) # Pastikan direktori 'scalers' ada
    filepath = os.path.join(scaler_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler disimpan ke: {filepath}")

@lru_cache(maxsize=1)
def load_scaler(filepath): # Parameter sudah filepath, jadi tidak perlu diubah
    """Memuat objek MinMaxScaler dari file pickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_encoder_info(encoder_info, filename='encoder_info.pkl'):
    """Menyimpan dictionary berisi informasi encoder (misal: kolom one-hot)."""
    encoder_dir = os.path.join(BASE_DIR, 'encoders')
    os.makedirs(encoder_dir, exist_ok=True) # Pastikan direktori 'encoders' ada
    filepath = os.path.join(encoder_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(encoder_info, f)
    print(f"Informasi encoder disimpan ke: {filepath}")
    
def load_encoder_info(filepath):
    """Memuat informasi encoder."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File informasi encoder tidak ditemukan: {filepath}")
    with open(filepath, 'rb') as f:
        info = pickle.load(f)
    print(f"Informasi encoder dimuat dari: {filepath}")
    return info

