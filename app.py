from flask import Flask
from train_model import train_bp
from predict import predict_bp

app = Flask(__name__)
app.register_blueprint(train_bp)
app.register_blueprint(predict_bp)

@app.route('/')
def index():
    return "API Prediksi Konsumsi Listrik RNN"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)