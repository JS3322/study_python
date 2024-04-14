import logging
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)

def log_activity(func):
    """Decorator to log information before and after a function is called."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') if 'request' in kwargs else None
        if request:
            logging.info(f"Handling {request.method} request to {request.url}")
        else:
            logging.info(f"Handling call to {func.__name__}")
        result = await func(*args, **kwargs)
        logging.info(f"Completed test Example {func.__name__} with result {result}")
        return result
    return wrapper