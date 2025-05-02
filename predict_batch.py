#predict_batch.py
from flask import Blueprint, request, jsonify
from services.predict_service import predict_batch_results
from helper.redis_helper import redis_client
import json

predict_batch_bp = Blueprint('predict_batch', __name__)

@predict_batch_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        required_fields = ['customer_ids', 'batch_id', 'jumlah_bulan']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        results = predict_batch_results(
            data['customer_ids'],
            data['batch_id'],
            data['jumlah_bulan']
        )
        
        return jsonify({
            'success': True,
            'processed': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@predict_batch_bp.route('/predict_batch/progress/<batch_id>', methods=['GET'])
def get_progress(batch_id):
    try:
        progress = redis_client.get(f"prediction:{batch_id}:progress")
        if progress:
            return jsonify(json.loads(progress)), 200
        return jsonify({'error': 'Progress not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500