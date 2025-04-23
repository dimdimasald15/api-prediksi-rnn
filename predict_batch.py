# predict_batch.py
from flask import Blueprint, request, jsonify
from services.predict_service import predict_customer

predict_batch_bp = Blueprint('predict_batch', __name__)

@predict_batch_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if not data or 'customers' not in data or 'jumlah_bulan' not in data:
            return jsonify({'error': 'Data harus mengandung "customers" dan "jumlah_bulan"'}), 400

        customer_list = data['customers']  # List of dicts: {customer_id, nama}
        jumlah_bulan = int(data['jumlah_bulan'])

        results = []
        for customer in customer_list:
            try:
                result = predict_customer(
                    customer_id=customer['customer']['customer_id'],
                    nama=customer['customer']['nama'],
                    jumlah_bulan=jumlah_bulan
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'customer_id': customer.get('customer_id'),
                    'error': str(e)
                })

        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': f'Error batch prediction: {str(e)}'}), 500
