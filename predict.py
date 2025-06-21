from flask import Blueprint, request, jsonify
from services.predict_service import predict_customer

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not request.is_json:
            return jsonify({'error': 'Request harus dalam format JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Body request kosong'}), 400
        
        customer_id = data.get('customer_id')
        if not customer_id:
            return jsonify({'error': 'customer_id tidak disediakan'}), 400
            
        jumlah_bulan = data.get('jumlah_bulan', 1)
        if not isinstance(jumlah_bulan, int) or jumlah_bulan < 1:
            return jsonify({'error': 'jumlah_bulan harus berupa bilangan bulat positif'}), 400

        result = predict_customer(
            customer_id=customer_id,
            jumlah_bulan=jumlah_bulan
        )

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': f'Error tidak terduga: {str(e)}'}), 500