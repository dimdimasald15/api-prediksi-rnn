# helper/redis_helper.py
import redis
import json
from functools import wraps
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisClient:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance
    
    def _init_client(self):
        try:
            # Konfigurasi dari environment variables
            self.client = redis.Redis(
                host='redis-12766.c61.us-east-1-3.ec2.redns.redis-cloud.com',
                port=12766,
                decode_responses=True,
                username="default",
                password="6bOyGhuYME9kUfyNFegzl8WDhkK4XPB5",
                socket_timeout=10,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            raise

    def cache_progress(self, key, data, ttl=3600):
        """Menyimpan data progress dengan TTL"""
        try:
            serialized = json.dumps(data)
            return self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache progress failed: {str(e)}")
            return False

    def get_progress(self, key):
        """Mengambil data progress"""
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Get progress failed: {str(e)}")
            return None

    def health_check(self):
        """Memeriksa kesehatan koneksi"""
        try:
            return self.client.ping()
        except:
            return False

# Singleton instance
redis_client = RedisClient().client

def redis_connection_required(func):
    """Decorator untuk memastikan koneksi Redis tersedia"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not redis_client or not redis_client.ping():
            raise ConnectionError("Redis connection is not available")
        return func(*args, **kwargs)
    return wrapper