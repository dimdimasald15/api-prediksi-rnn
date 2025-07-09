# utils.py
import pickle
import os
from sqlalchemy import create_engine
from functools import lru_cache

# Global constants
MODEL_PATH = 'model_rnn_konsumsi.keras'
X_SCALER_PATH = 'scaler.pkl'
Y_SCALER_PATH = 'scaler_y.pkl'
PLOT_FOLDER = 'static/plots'

# Database config
db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'port': 3306,
    'database': 'dbprediction'
}

def get_db_connection():
    return create_engine(
        f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}",
        connect_args={"connect_timeout": 20},
        pool_pre_ping=True
    )

def save_scaler(scaler, path='scaler.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

@lru_cache(maxsize=1)
def load_scaler(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' tidak ditemukan. Silakan latih ulang model terlebih dahulu.")
    with open(path, 'rb') as f:
        return pickle.load(f)
