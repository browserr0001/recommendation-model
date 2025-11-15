from flask import Flask, Response, jsonify
from functools import wraps
from collections import defaultdict
import os, json, time, threading
from datetime import datetime

# Import model loading functions
from app.model_loader import (
    get_model,
    load_model_from_path,
    get_current_model_and_path,
    set_current_model,
)

app = Flask(__name__)
MODEL_TAG = os.getenv('MODEL_TAG', "Updated_Model") 
if MODEL_TAG not in ["Initial_Model", "Updated_Model"]:
    MODEL_TAG = "Updated_Model"
model = get_model(tag=MODEL_TAG)

# Get backend container name for logging
BACKEND_NAME = os.environ.get("BACKEND_NAME", "unknown-backend")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'pkl_models')
ACTIVE_FILENAME = f'content_based_model_{MODEL_TAG}.pkl'
ACTIVE_PATH = os.path.join(MODELS_DIR, ACTIVE_FILENAME)
print(f"[{BACKEND_NAME}] Active model file: {ACTIVE_PATH}", flush=True)

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
        'model_metadata': prediction_metadata, 
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

    # get the current model and path
    local_model = model

    # Get recommendations with metadata
    recs, inf_time, prediction_metadata = local_model.get_recommendations(user_id, top_n=20)
    result = ','.join(str(i) for i in recs)

    # Store inference time
    inference_time = inf_time

    # Log prediction with full provenance
    log_prediction(user_id, recs, prediction_metadata, inf_time)

    return Response(result, mimetype='text/plain')


@app.route('/metrics', methods=['GET'])
def model_info():
    current_model, current_path = get_current_model_and_path()
    metadata = current_model.get_metadata()
    metadata.update({
        "backend": BACKEND_NAME,
        "model_path": current_path,
        "model_tag": MODEL_TAG,
    })
    return jsonify(metadata)

# @app.route('/admin/reload', methods=['POST'])
# def admin_reload():
#     _swap_model()
#     return jsonify({"ok": True, "reloaded": ACTIVE_PATH})

# Hot reload when content_based_model_<TAG>.pkl changes

def _swap_model():
    global model
    try:
        new_model = load_model_from_path(ACTIVE_PATH)
        set_current_model(new_model, ACTIVE_PATH)
        model = new_model
        print(f"[{BACKEND_NAME}] Hot-reloaded model: {ACTIVE_PATH}", flush=True)
    except Exception as e:
        print(f"[{BACKEND_NAME}] Reload failed: {e}", flush=True)

def start_model_watcher():
    """Start watchdog or a polling thread to hot-reload the active model."""
    use_polling = os.getenv("WATCHDOG_FORCE_POLLING", "0") == "1"

    if not use_polling:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def on_any_event(self, event):
                    if event.is_directory:
                        return
                    if os.path.basename(event.src_path) == ACTIVE_FILENAME:
                        time.sleep(0.05)  # let atomic copy finish
                        _swap_model()

            observer = Observer()
            observer.schedule(_Handler(), path=MODELS_DIR, recursive=False)
            observer.daemon = True
            observer.start()
            print(f"[{BACKEND_NAME}] Watching {MODELS_DIR} for {ACTIVE_FILENAME}", flush=True)
            return  # success, using watchdog
        except Exception as e:
            print(f"[{BACKEND_NAME}] Watchdog init failed ({e}); falling back to polling.", flush=True)

    try:
        st = os.stat(ACTIVE_PATH)
        last_sig = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
    except FileNotFoundError:
        last_sig = None

    interval = float(os.getenv("MODEL_WATCH_INTERVAL", "1.0"))
    print(f"[{BACKEND_NAME}] Using polling for {ACTIVE_FILENAME} (interval={interval}s)", flush=True)


    def _poller():
        nonlocal last_sig
        while True:
            try:
                st = os.stat(ACTIVE_PATH)
                sig = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
                if sig != last_sig:
                    last_sig = sig
                    _swap_model()
            except FileNotFoundError:
                # file temporarily missing? keep waiting
                pass
            except Exception as _e:
                print(f"[{BACKEND_NAME}] Poller error: {_e}", flush=True)
            time.sleep(interval)

    threading.Thread(target=_poller, daemon=True).start()

if __name__ == '__main__':
    start_model_watcher()
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
