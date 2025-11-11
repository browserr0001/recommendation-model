from flask import Flask, Response
import os
import itertools
import requests

app = Flask(__name__)

# Comma-separated backend URLs such as "http://backend-a:5000,http://backend-b:5000"
BACKENDS = [b for b in os.environ.get("BACKENDS", "http://backend-a:5000").split(",") if b]

# Round-robin iterator over the backend list
server_pool = itertools.cycle(BACKENDS)


def choose_backend() -> str:
    """
    Pure round-robin:
      - First request -> first backend
      - Second request -> second backend
      - Third request -> first backend
      ...
    """
    return next(server_pool)


@app.get("/recommend/<userid>")
def recommend(userid):
    backend = choose_backend()

    try:
        resp = requests.get(f"{backend}/recommend/{userid}", timeout=0.55)
    except requests.RequestException as e:
        return Response(f"Backend error: {e}", status=502, mimetype="text/plain")

    return Response(
        resp.content,
        status=resp.status_code,
        mimetype=resp.headers.get("Content-Type", "text/plain") or "text/plain",
    )


@app.get("/api")
def api_proxy():
    backend = choose_backend()
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
