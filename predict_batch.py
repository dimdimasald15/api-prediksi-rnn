#predict_batch.py
from flask import Blueprint, request, jsonify
from services.predict_service import predict_batch_results
from helper.redis_helper import redis_client
import json
import logging

predict_batch_bp = Blueprint('predict_batch', __name__)

@predict_batch_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        required_fields = ['customer_ids', 'batch_id', 'jumlah_bulan']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        # Check if customer_ids is a list and not empty
        if not isinstance(data['customer_ids'], list) or not data['customer_ids']:
            return jsonify({'error': 'customer_ids must be a non-empty list'}), 400
        
        # Validate jumlah_bulan is a positive integer
        try:
            jumlah_bulan = int(data['jumlah_bulan'])
            if jumlah_bulan <= 0:
                raise ValueError("jumlah_bulan must be positive")
        except (ValueError, TypeError):
            return jsonify({'error': 'jumlah_bulan must be a positive integer'}), 400
        
        results = predict_batch_results(
            data['customer_ids'],
            data['batch_id'],
            data['jumlah_bulan']
        )
        
        return jsonify(results), 200
        
    except Exception as e:
        logging.error(f"Batch prediction API error: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@predict_batch_bp.route('/predict_batch/progress/<batch_id>', methods=['GET'])
def get_progress(batch_id):
    try:
        progress_key = f"prediction:{batch_id}:progress"
        progress = redis_client.get(progress_key)
        
        if progress:
            return jsonify(json.loads(progress)), 200
        return jsonify({'error': 'Progress not found', 'batch_id': batch_id}), 404
    except Exception as e:
        logging.error(f"Get progress API error: {str(e)}")
        return jsonify({'error': str(e)}), 500