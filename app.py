from flask import Flask, jsonify, send_from_directory
import os
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
    "progress": "0%"
}

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
def update_progress(progress):
    global training_status
    training_status["progress"] = f"{progress}%"
    print(f"Progress updated: {progress}%")

def train_model_thread():
    global training_status, model

    training_status.update({
        "is_training": True,
        "status": "training",
        "error": None,
        "progress": "0%"
    })

    def set_progress(persen):
        training_status["progress"] = f"{persen}%"

    try:
        from train_model import train_and_save_model
        train_and_save_model(progress_callback=set_progress)

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

@app.route('/training-status', methods=['GET'])
def get_training_status():
    global training_status
    return jsonify(training_status)

@app.route('/plot/<filename>', methods=['GET'])
def get_plot(filename):
    return send_from_directory(PLOT_FOLDER, filename)

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
    app.run(host='0.0.0.0', port=8000)