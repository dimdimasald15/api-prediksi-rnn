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





- Golongan R-1/TR daya 900 VA, Rp 1.352 per kWh
- Golongan R-1/ TR daya 1.300 VA, Rp 1.444,70 per kWh
- Golongan R-1/ TR daya 2.200 VA, Rp 1.444,70 per kWh
- Golongan R-2/ TR daya 3.500-5.500 VA, Rp 1.699,53 per kWh
- Golongan R-3/ TR daya 6.600 VA ke atas, Rp 1.699,53 per kWh
- Golongan B-2/ TR daya 6.600 VA-200 kVA, Rp 1.444,70 per kWh
- Golongan B-3/ Tegangan Menengah (TM) daya di atas 200 kVA, Rp 1.114,74 per kWh
- Golongan I-3/ TM daya di atas 200 kVA, Rp 1.114,74 per kWh
- Golongan I-4/ Tegangan Tinggi (TT) daya 30.000 kVA ke atas, Rp 996,74 per kWh
- Golongan P-1/ TR daya 6.600 VA-200 kVA, Rp 1.699,53 per kWh
- Golongan P-2/ TM daya di atas 200 kVA, Rp 1.522,88 per kWh
- Golongan P-3/ TR untuk penerangan jalan umum,
- Golongan L/ TR, TM, TT, Rp 1.644,52 per kWh.
