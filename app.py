from flask import Flask
import os
from train_model import train_bp
from predict import predict_bp

app = Flask(__name__)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)

model_path = 'model_rnn_konsumsi.keras'

def load_model():
    global model
    model = tf.keras.models.load_model(model_path)

if os.path.exists(model_path):
    load_model()

@app.route('/')
def index():
    status = "tersedia" if model else "tidak tersedia"
    return f"API Prediksi Konsumsi Listrik RNN (Model {status})"

@app.route('/retrain', methods=['POST'])  # opsional route lain
def retrain():
    try:
        from train_model import train_and_save_model
        train_and_save_model()
        load_model()
        return {"message": "Model retrained and saved successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)