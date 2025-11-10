from flask import Flask, Response, request, jsonify
from app.model_loader import get_model
# from model.src.content_based import get_user_metadata
from functools import wraps
from collections import defaultdict
import time
import os

app = Flask(__name__)
model = get_model()

# Get backend container name for logging
BACKEND_NAME = os.environ.get("BACKEND_NAME", "unknown-backend")

# For logging
# For logging - use global counters instead of per-endpoint
total_requests = 0
inference_time = 0
get_user_metadata_calls = 0
users_in_cache_hits = 0
user_id_log = 0

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global total_requests
        
        # Start timing the request
        start_time = time.time()
        
        # Increment global request counter
        total_requests += 1

        # Call the original view function
        response = f(*args, **kwargs)
        
        # Calculate request duration
        request_duration = time.time() - start_time
        
        # Normalize endpoint for logging (remove user ID)
        endpoint = request.path
        if endpoint.startswith('/recommend/'):
            endpoint = '/recommend/<userid>'
        
        # Enhanced logging output with global counters
        print(f"[{endpoint}] Total Requests: {total_requests}, "
              f"Backend: {BACKEND_NAME}, "
              f"Request Time: {request_duration:.3f}s, "
              f"Inference time: {inference_time:.3f}s, "
              f"user_id: {user_id_log}"
            #   f"get_user_metadata calls: {get_user_metadata_calls}/{total_requests} ({get_user_metadata_calls/total_requests*100:.1f}%), "
            #   f"Users in cache hits: {users_in_cache_hits}/{total_requests} ({users_in_cache_hits/total_requests*100:.1f}%)"
            )

        return response
    return decorated_function



@app.route('/api', methods=['GET'])
def test():
    return Response(f"Movie Recommendation API is running on {BACKEND_NAME}.", mimetype='text/plain')


@app.route('/recommend/<userid>', methods=['GET'])
@log_request
def recommend(userid):
    global get_user_metadata_calls, users_in_cache_hits, inference_time, user_id_log
    
    try:
        user_id = int(userid)
    except Exception:
        user_id = userid  # fallback to string
    

    user_id_log = user_id

    recs, inf_time = model.get_recommendations(user_id, top_n=20)
    result = ','.join(str(i) for i in recs)

    # Store inference time
    inference_time = inf_time

    return Response(result, mimetype='text/plain')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
