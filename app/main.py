from flask import Flask, Response, request, jsonify
from app.model_loader import get_model
# from model.src.content_based import get_user_metadata
from functools import wraps
from collections import defaultdict
import time
import json
import os
from datetime import datetime


app = Flask(__name__)
MODEL_TAG = os.getenv('MODEL_TAG', "Updated_Model") 
model = get_model(tag=MODEL_TAG)

# Get backend container name for logging
BACKEND_NAME = os.environ.get("BACKEND_NAME", "unknown-backend")

# For logging
total_requests = 0
inference_time = 0
get_user_metadata_calls = 0
users_in_cache_hits = 0
user_id_log = 0


# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

def log_prediction(user_id, recommendations, prediction_metadata, inference_time):
    """Log prediction with full provenance information"""
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'recommendations': recommendations,
        'inference_time': inference_time,
        'model_metadata': prediction_metadata
    }
    
    # Append to daily log file
    log_file = os.path.join(LOGS_DIR, f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl")
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return log_entry

@app.route('/api', methods=['GET'])
def test():
    return Response(f"Movie Recommendation API is running on {BACKEND_NAME}.", mimetype='text/plain')


@app.route('/recommend/<userid>', methods=['GET'])
def recommend(userid):
    global get_user_metadata_calls, users_in_cache_hits, inference_time, user_id_log
    
    try:
        user_id = int(userid)
    except Exception:
        user_id = userid  # fallback to string
    
    user_id_log = user_id

    # Get recommendations with metadata
    recs, inf_time, prediction_metadata = model.get_recommendations(user_id, top_n=20)
    result = ','.join(str(i) for i in recs)

    # Store inference time
    inference_time = inf_time
    
    # Log prediction with full provenance
    log_prediction(user_id, recs, prediction_metadata, inf_time)

    return Response(result, mimetype='text/plain')


@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get model metadata and provenance information"""
    metadata = model.get_metadata()
    return jsonify(metadata)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
