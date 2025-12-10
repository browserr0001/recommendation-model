from flask import Flask, Response
import os
import requests

app = Flask(__name__)

# Comma-separated backend URLs such as "http://backend-a:5001,http://backend-b:5001"
BACKENDS = [b for b in os.environ.get("BACKENDS", "http://backend-a:5001").split(",") if b]
# print(BACKENDS, flush=True)

def choose_backend(userid: str) -> str:
    """
    Route based on user ID:
      - Odd user IDs -> first backend (backend-a)
      - Even user IDs -> second backend (backend-b)
    """
    try:
        # Extract numeric part if userid contains "userid=" prefix
        userid_str = userid.split('=')[-1] if '=' in userid else userid
        user_id_int = int(userid_str)
        # Odd goes to backend-a (index 0), even goes to backend-b (index 1)
        backend_index = 0 if user_id_int % 2 == 1 else 1
        # Handle case where only one backend is configured
        if backend_index >= len(BACKENDS):
            backend_index = 0

        print(f"Routing user_id {user_id_int} to backend index {backend_index} ({BACKENDS[backend_index]})", flush=True)
        return BACKENDS[backend_index]
    except (ValueError, IndexError) as e:
        # Fallback to first backend if userid is not a valid integer
        print(f"Error choosing backend for userid '{userid}': {e}", flush=True)
        return BACKENDS[0] if BACKENDS else "http://backend-a:5001"


@app.get("/recommend/<userid>")
def recommend(userid):
    backend = choose_backend(userid)

    try:
        resp = requests.get(f"{backend}/recommend/{userid}", timeout=0.55)
    except requests.RequestException as e:
        return Response(f"Backend error: {e}", status=502, mimetype="text/plain")

    return Response(
        resp.content,
        status=resp.status_code,
        mimetype=resp.headers.get("Content-Type", "text/plain") or "text/plain",
    )


@app.get("/metrics")
def model_info():
    backend = BACKENDS[0] if BACKENDS else "http://backend-a:5001"
    try:
        resp = requests.get(f"{backend}/metrics", timeout=0.55)
    except requests.RequestException as e:
        return Response(f"Backend error: {e}", status=502, mimetype="text/plain")

    return Response(
        resp.content,
        status=resp.status_code,
        mimetype=resp.headers.get("Content-Type", "text/plain") or "text/plain",
    )

@app.get("/api")
def api_proxy():
    backend = BACKENDS[0] if BACKENDS else "http://backend-a:5001"
    try:
        resp = requests.get(f"{backend}/api", timeout=0.6)
    except requests.RequestException as e:
        return Response(f"Backend error: {e}", status=502, mimetype="text/plain")

    return Response(
        resp.content,
        status=resp.status_code,
        mimetype=resp.headers.get("Content-Type", "text/plain") or "text/plain",
    )


@app.get("/health")
def health():
    return {"status": "ok", "backends": BACKENDS}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=False)
