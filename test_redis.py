from flask import Blueprint, request, jsonify
from helper.redis_helper import redis_client, RedisClient
import json

test_redis_bp = Blueprint('test_redis', __name__)

@test_redis_bp.route('/test_redis', methods=['GET'])
def test_redis():
    test_results = {
        'singleton_test': False,
        'connection_test': False,
        'progress_tracking_test': False,
        'all_passed': False
    }
    
    try:
        # Test singleton
        instance1 = RedisClient()
        instance2 = RedisClient()
        test_results['singleton_test'] = (instance1 is instance2)
        
        # Test connection
        test_results['connection_test'] = redis_client.ping()
        
        # Test progress tracking
        test_key = "test:progress:123"
        test_data = {'processed': 50, 'total': 100}
        
        set_result = redis_client.setex(
            test_key,
            60,  # 1 menit expiry
            json.dumps(test_data)
        )
        
        retrieved = redis_client.get(test_key)
        retrieved_data = json.loads(retrieved) if retrieved else None
        
        test_results['progress_tracking_test'] = (
            set_result and 
            retrieved_data == test_data
        )
        
        # Hapus test key
        redis_client.delete(test_key)
        
        # Update overall status
        test_results['all_passed'] = all([
            test_results['singleton_test'],
            test_results['connection_test'],
            test_results['progress_tracking_test']
        ])
        
        status_code = 200 if test_results['all_passed'] else 500
        return jsonify(test_results), status_code
        
    except Exception as e:
        test_results['error'] = str(e)
        return jsonify(test_results), 500