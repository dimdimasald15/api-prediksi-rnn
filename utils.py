# utils.py
import pickle
import os
from sqlalchemy import create_engine

# Global constants
MODEL_PATH = 'model_rnn_konsumsi.keras'
X_SCALER_PATH = 'scaler.pkl'
Y_SCALER_PATH = 'scaler_y.pkl'
PLOT_FOLDER = 'static/plots'

# Database config
db_config = {
    'user': 'root',
    'password': 'ThpPEjAeCkstBjBiUdtmqcqwYRGhbyKh',
    'host': 'switchback.proxy.rlwy.net',
    'port': 53354,
    'database': 'railway'
}

def get_db_connection():
    return create_engine(
        "mysql+mysqlconnector://root:ThpPEjAeCkstBjBiUdtmqcqwYRGhbyKh@switchback.proxy.rlwy.net:53354/railway"
    )

def save_scaler(scaler, path='scaler.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
    with open(path, 'rb') as f:
        return pickle.load(f)

# def load_y_scaler(path='scaler_y.pkl'):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Y Scaler file '{path}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
#     with open(path, 'rb') as f:
#         return pickle.load(f)

