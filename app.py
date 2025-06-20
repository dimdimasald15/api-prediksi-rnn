import logging
from flask import Flask, jsonify, send_from_directory, request
import os
import json
import threading
import tensorflow as tf
from train_model import train_bp
from predict import predict_bp
from predict_batch import predict_batch_bp
from utils import PLOT_FOLDER


app = Flask(__name__)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(predict_batch_bp)

model = None
model_path = 'model_rnn_konsumsi.keras'

# Status pelatihan global
training_status = {
    "is_training": False,
    "status": "idle",
    "error": None,
     "progress": "0%",
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": None,
    "current_val_loss": None
}


# Konfigurasi logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
def update_training_status_callback(
    progress_percent,
    epoch=None,
    total_epochs=None,
    loss=None,
    val_loss=None
):
    global training_status
    
    training_status["progress"] = f"{progress_percent}%"
    
    if epoch is not None:
        training_status["current_epoch"] = epoch
    if total_epochs is not None:
        training_status["total_epochs"] = total_epochs
    if loss is not None:
        training_status["current_loss"] = float(loss) # Konversi ke float untuk JSON
    if val_loss is not None:
        training_status["current_val_loss"] = float(val_loss) # Konversi ke float untuk JSON
        
    loss_str = f"{training_status['current_loss']:.4f}" if training_status['current_loss'] is not None else "N/A"
    val_loss_str = f"{training_status['current_val_loss']:.4f}" if training_status['current_val_loss'] is not None else "N/A"

    print(f"Training Progress: {training_status['progress']} | "
          f"Epoch: {training_status['current_epoch']}/{training_status['total_epochs']} | "
          f"Loss: {loss_str} | "
          f"Val Loss: {val_loss_str}")


def train_model_thread():
    global training_status, model

    training_status.update({
        "is_training": True,
        "status": "training",
        "error": None,
        "progress": "0%",
        "current_epoch": 0, # Reset saat training dimulai
        "total_epochs": 0,  # Akan diisi dari train_and_save_model
        "current_loss": None,
        "current_val_loss": None
    })

    def set_progress(persen):
        training_status["progress"] = f"{persen}%"

    try:
        from train_model import train_and_save_model
        train_and_save_model(progress_callback=update_training_status_callback)

        if load_model():
            training_status["status"] = "completed"
        else:
            training_status["status"] = "completed_with_errors"
            training_status["error"] = "Model berhasil dilatih tetapi gagal dimuat"
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in training thread: {str(e)}")
        print(error_trace)
        logging.error(f"Error in training thread: {str(e)}\n{error_trace}") # Log error lengkap
        training_status["status"] = "failed"
        training_status["error"] = str(e)
    finally:
        training_status["is_training"] = False

# Load model saat startup jika ada
if os.path.exists(model_path):
    load_model()

@app.route('/')
def index():
    status = "tersedia" if model else "tidak tersedia"
    training_info = f" (sedang dilatih)" if training_status["is_training"] else ""
    return f"API Prediksi Konsumsi Listrik RNN (Model {status}{training_info})"

@app.route('/training-model', methods=['POST'])
def train_model():
    global training_status
    
    # Cek apakah model sedang dilatih
    if training_status["is_training"]:
        return jsonify({
            "message": "Model sedang dilatih", 
            "status": training_status["status"]
        }), 409  # 409 Conflict
    
    # Mulai thread pelatihan
    thread = threading.Thread(target=train_model_thread)
    thread.daemon = True  # Thread akan dihentikan saat program utama selesai
    thread.start()
    
    return jsonify({
        "message": "Pelatihan model dimulai di background",
        "status": training_status["status"]
    }), 202  # 202 Accepted

@app.route('/model-metrics', methods=['GET']) # Tambahkan ini jika belum ada
def get_model_metrics():
    metrics_file_path = os.path.join(PLOT_FOLDER, 'train_model/model_metrics.json')
    if os.path.exists(metrics_file_path):
        try:
            with open(metrics_file_path, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to read model metrics file."}), 500
    else:
        return jsonify({"message": "Model metrics not found. Train the model first."}), 404


@app.route('/training-status', methods=['GET'])
def get_training_status():
    global training_status
    return jsonify(training_status)

@app.route('/plot/<filename>', methods=['GET'])
def get_plot(filename):
    try:
        return send_from_directory(PLOT_FOLDER, filename)
    except FileNotFoundError:
        logging.error(f"File not found: {filename} - Request from {request.remote_addr}")
        return "File not found", 404
    except Exception as e:
        logging.exception(f"Error during file request: {e} - Request from {request.remote_addr}") # Log detail error
        return "Internal Server Error", 500

@app.route('/delete-plot/<filename>', methods=['DELETE'])
def delete_plot(filename):
    file_path = os.path.join(PLOT_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"message": "File tidak ditemukan"}), 404

    try:
        os.remove(file_path)
        return jsonify({"message": f"File {filename} berhasil dihapus"}), 200
    except Exception as e:
        return jsonify({"message": "Gagal menghapus file", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)