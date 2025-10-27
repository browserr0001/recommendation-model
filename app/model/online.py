from confluent_kafka import Consumer
from collections import defaultdict, deque
import re
import time
import json
import os
import numpy as np
from datetime import datetime
from dateutil import parser
import pandas as pd
import csv
from pathlib import Path

# ---------------- CONFIGURATION ----------------------

KAFKA_BROKER = "localhost:9092"
TOPIC = "movielog8"
GROUP_ID = "online-eval"
PRINT_INTERVAL = 20        # seconds between metric updates
WINDOW_SIZE = 3000         # number of recent recs to keep
CTR_LOOKAHEAD = 1800       # seconds (30-min window for CTR)
OUTPUT_JSON = "metrics/online_evaluation_live.json"

MIN_RATINGS = 5            # threshold for warm-start
MIN_MINUTES = 30           # threshold for warm-start
CATALOG_SIZE = 1000        # approximate unique movies

# ---------------- PATTERN DEFINITIONS ----------------

rec_pattern = re.compile(r'^(.*?),(\d+),recommendation request.*?result: (.+?), (\d+) ms')
watch_pattern = re.compile(r'^(.*?),(\d+),GET /data/m/(.+?)/(\d+)\.mpg')
rating_pattern = re.compile(r'^(.*?),(\d+),GET /rate/(.+)=(\d+)')

# ---------------- STATE STORAGE ----------------------

recommendations = deque(maxlen=WINDOW_SIZE)   # (ts, user_id, recs)
user_watched = defaultdict(list)
user_rated = defaultdict(list)
user_rating_counts = defaultdict(int)
user_minutes_watched = defaultdict(int)
rating_sum, rating_count = 0, 0

COHORTS = ["all", "warm", "cold"]
metrics_buffer = {c: defaultdict(list) for c in COHORTS}

# Try to load model-known users from training data (optional)
MODEL_KNOWN_USERS = set()
try:
    ratings = pd.read_parquet("data-pull/data/ratings/ratings.parquet")
    MODEL_KNOWN_USERS = set(ratings["user_id"].unique())
    print(f"Loaded {len(MODEL_KNOWN_USERS)} model-known users.")
except Exception as e:
    print("Could not load training user IDs:", e)

# ---------------- HELPER FUNCTIONS -------------------

def parse_time(ts: str) -> int:
    try:
        return int(parser.parse(ts).timestamp())
    except Exception:
        return int(time.time())


def dcg_at_k(hits):
    return sum(1.0 / np.log2(i + 2) for i, h in enumerate(hits) if h)


def cohort_for_rec(user_id: int) -> str:
    """Decide whether a user is warm or cold at rec time."""
    if MODEL_KNOWN_USERS and user_id in MODEL_KNOWN_USERS:
        return "warm"
    if user_rating_counts[user_id] >= MIN_RATINGS or user_minutes_watched[user_id] >= MIN_MINUTES:
        return "warm"
    return "cold"


def process_line(line: str):
    """Parse Kafka line and update user state."""
    global rating_sum, rating_count

    # --- Recommendation Event ---
    rec_match = rec_pattern.match(line)
    if rec_match:
        ts, user_id, recs, latency = rec_match.groups()
        uid = int(user_id)
        rec_time = parse_time(ts)
        rec_list = [m.strip() for m in recs.split(',') if m.strip()]
        rt_ms = int(latency)

        recommendations.append((rec_time, uid, rec_list))
        cohort = cohort_for_rec(uid)

        # Save latency
        for c in [cohort, "all"]:
            metrics_buffer[c]["inference_time"].append(rt_ms)

        evaluate_user(uid, rec_list, cohort, rec_time)
        return

    # --- Watch Event ---
    watch_match = watch_pattern.match(line)
    if watch_match:
        ts, user_id, movie_id, minute = watch_match.groups()
        t = parse_time(ts)
        uid = int(user_id)
        user_watched[uid].append((t, movie_id))
        user_minutes_watched[uid] += 1
        return

    # --- Rating Event ---
    rating_match = rating_pattern.match(line)
    if rating_match:
        ts, user_id, movie_id, rating = rating_match.groups()
        t = parse_time(ts)
        uid = int(user_id)
        r = int(rating)
        user_rated[uid].append((t, movie_id, r))
        user_rating_counts[uid] += 1
        rating_sum += r
        rating_count += 1
        return


def evaluate_user(user_id: int, recs: list, cohort: str, rec_time: int, k: int = 20):
    """Compute Precision, Recall, NDCG, HitRate per user."""
    watched_after = [m for t, m in user_watched[user_id] if t >= rec_time]
    if not watched_after:
        return

    hits = [1 if m in watched_after else 0 for m in recs[:k]]
    n_hits = sum(hits)
    denom = max(1, len(set(watched_after)))

    precision = n_hits / k
    recall = n_hits / denom
    ndcg = dcg_at_k(hits) / dcg_at_k(sorted(hits, reverse=True)) if n_hits > 0 else 0.0
    hit_rate = 1.0 if n_hits > 0 else 0.0

    for c in [cohort, "all"]:
        metrics_buffer[c]["precision"].append(precision)
        metrics_buffer[c]["recall"].append(recall)
        metrics_buffer[c]["ndcg"].append(ndcg)
        metrics_buffer[c]["hit_rate"].append(hit_rate)


def compute_ctr_by_cohort(now: int):
    out = {}
    for c in COHORTS:
        total, clicked = 0, 0
        for rec_time, uid, recs in list(recommendations):
            if now - rec_time > CTR_LOOKAHEAD:
                continue
            if c != "all" and cohort_for_rec(uid) != c:
                continue
            w = [m for t, m in user_watched[uid] if 0 <= t - rec_time <= CTR_LOOKAHEAD]
            if w and set(recs) & set(w):
                clicked += 1
            total += 1
        ctr = clicked / total if total > 0 else 0.0
        metrics_buffer[c]["ctr"].append(ctr)
        out[c] = (ctr, total, clicked)
    return out


def diversity_coverage_by_cohort():
    out = {}
    for c in COHORTS:
        rec_items = []
        for rec_time, uid, recs in list(recommendations):
            if c != "all" and cohort_for_rec(uid) != c:
                continue
            rec_items.extend(recs)
        uniq = len(set(rec_items))
        total = max(1, len(rec_items))
        coverage = uniq / total
        diversity = uniq / CATALOG_SIZE
        metrics_buffer[c]["diversity"].append(diversity)
        metrics_buffer[c]["coverage"].append(coverage)
        out[c] = (diversity, coverage)
    return out


def aggregate_metrics_all():
    def avg(buf, key):
        return round(float(np.mean(buf[key])), 4) if buf[key] else 0.0

    def inf_stats(buf):
        arr = buf["inference_time"]
        if not arr:
            return (0.0, 0.0)
        return round(float(np.mean(arr)), 2), round(float(np.std(arr)), 2)

    avg_rating = (rating_sum / rating_count) if rating_count > 0 else 0.0
    payload = {}
    for c in COHORTS:
        buf = metrics_buffer[c]
        avg_inf, std_inf = inf_stats(buf)
        payload[c] = {
            "precision@K": avg(buf, "precision"),
            "recall@K": avg(buf, "recall"),
            "ndcg@K": avg(buf, "ndcg"),
            "hit_rate@K": avg(buf, "hit_rate"),
            "ctr": avg(buf, "ctr"),
            "diversity": avg(buf, "diversity"),
            "coverage": avg(buf, "coverage"),
            "avg_inference_time_ms": avg_inf,
            "std_inference_time_ms": std_inf,
        }

    payload["avg_rating"] = round(avg_rating, 3)
    payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["users_tracked"] = len(user_watched)
    return payload

# --------------- PERSISTENCE FUNCTION ----------------

def persist_metrics(metrics):
    """Append current metrics snapshot to JSON and CSV history files."""
    timestamp = metrics.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    json_path = Path("metrics/online_evaluation_history.json")
    csv_path = Path("metrics/online_evaluation_history.csv")

    # --- JSON append ---
    history = []
    if json_path.exists():
        try:
            history = json.loads(json_path.read_text())
        except Exception:
            pass
    history.append(metrics)
    json_path.write_text(json.dumps(history, indent=2))

    # --- CSV append ---
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "cohort", "precision@K", "recall@K", "ctr", "diversity", "coverage"])
        for cohort in ["all", "warm", "cold"]:
            vals = metrics[cohort]
            writer.writerow([
                timestamp, cohort,
                vals["precision@K"], vals["recall@K"],
                vals["ctr"], vals["diversity"], vals["coverage"]
            ])

# ---------------- STREAMING LOOP ---------------------

def stream_and_analyze():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    conf = {"bootstrap.servers": KAFKA_BROKER,
            "group.id": GROUP_ID,
            "auto.offset.reset": "latest"}
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])

    print(f"Connected to Kafka topic '{TOPIC}' on {KAFKA_BROKER}")
    print("Starting real-time online evaluation... (Ctrl+C to stop)\n")
    last_print = time.time()

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue

            process_line(msg.value().decode("utf-8"))
            now = time.time()

            if now - last_print >= PRINT_INTERVAL:
                last_print = now
                ctrs = compute_ctr_by_cohort(now)
                divcov = diversity_coverage_by_cohort()
                metrics = aggregate_metrics_all()

                os.system("clear")
                print("=" * 75)
                print("Real-Time Online Evaluation — Warm vs Cold Cohorts")
                print("=" * 75)
                print(json.dumps(metrics, indent=2))
                print()
                for c in COHORTS:
                    ctr, total, clicks = ctrs[c]
                    print(f"{c.upper():<6} | CTR={ctr:.3f} | Recs={total} | Clicks={clicks}")
                print(f"Users tracked: {metrics['users_tracked']}\n")

                with open(OUTPUT_JSON, "w") as f:
                    json.dump(metrics, f, indent=2)

                # Persist metrics to history for reviewability
                persist_metrics(metrics)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        consumer.close()
        print("Kafka connection closed.")


if __name__ == "__main__":
    stream_and_analyze()
