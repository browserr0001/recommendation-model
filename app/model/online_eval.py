"""
Purpose:
--------
This script listens to live Kafka events (topic `movielog8`) in real time and
computes both traditional recommender metrics (precision, recall, ndcg, hit-rate,
diversity, coverage) and online-specific metrics (CTR, average rating, latency).
"""

from confluent_kafka import Consumer
from collections import defaultdict, deque
import re
import time
import json
import os
import numpy as np
from datetime import datetime
from dateutil import parser


KAFKA_BROKER = "localhost:9092"
TOPIC = "movielog8"
GROUP_ID = "online-eval"
PRINT_INTERVAL = 20        # seconds between metric updates
WINDOW_SIZE = 3000         # number of recent recommendations tracked
CTR_LOOKAHEAD = 1800       # seconds (30 min window for CTR)
OUTPUT_JSON = "metrics/online_evaluation_live.json"


rec_pattern = re.compile(r'^(.*?),(\d+),recommendation request.*?result: (.+?), \d+ ms')
watch_pattern = re.compile(r'^(.*?),(\d+),GET /data/m/(.+?)/(\d+)\.mpg')
rating_pattern = re.compile(r'^(.*?),(\d+),GET /rate/(.+)=(\d+)')


recommendations = deque(maxlen=WINDOW_SIZE)  # (timestamp, user_id, recs)
user_watched = defaultdict(list)             # user_id → [(timestamp, movie_id)]
user_rated = defaultdict(list)               # user_id → [(timestamp, movie_id, rating)]
metrics_buffer = defaultdict(list)           # rolling metrics history
rating_sum, rating_count = 0, 0


def parse_time(ts: str) -> int:
    """Safely parse timestamp from log line into epoch seconds."""
    try:
        return int(parser.parse(ts).timestamp())
    except Exception:
        return int(time.time())


def dcg_at_k(hits):
    """Compute Discounted Cumulative Gain (DCG) for ranking evaluation."""
    return sum(1.0 / np.log2(i + 2) for i, h in enumerate(hits) if h)


def process_line(line: str):
    """Parse a Kafka log line and update corresponding state."""
    global rating_sum, rating_count

    rec_match = rec_pattern.match(line)
    response_time_match = re.search(r'(\d+)\s*ms', line)
    if response_time_match:
        rt_ms = int(response_time_match.group(1))
        metrics_buffer["inference_time"].append(rt_ms)
    if rec_match:
        ts, user_id, recs = rec_match.groups()
        user_id = int(user_id)
        rec_time = parse_time(ts)
        rec_list = [m.strip() for m in recs.split(',') if m.strip()]
        recommendations.append((rec_time, user_id, rec_list))
        evaluate_user(user_id, rec_list)
        return

    watch_match = watch_pattern.match(line)
    if watch_match:
        ts, user_id, movie_id, minute = watch_match.groups()
        user_watched[int(user_id)].append((parse_time(ts), movie_id))
        return

    rating_match = rating_pattern.match(line)
    if rating_match:
        ts, user_id, movie_id, rating = rating_match.groups()
        user_id = int(user_id)
        r = int(rating)
        user_rated[user_id].append((parse_time(ts), movie_id, r))
        rating_sum += r
        rating_count += 1
        return


def evaluate_user(user_id: int, recs: list, k: int = 20):
    """Compute per-user Precision, Recall, NDCG, and Hit Rate."""
    watched_movies = [m for _, m in user_watched[user_id]]
    if not watched_movies:
        return

    hits = [1 if m in watched_movies else 0 for m in recs[:k]]
    n_hits = sum(hits)

    precision = n_hits / k if k > 0 else 0.0
    recall = n_hits / len(set(watched_movies)) if watched_movies else 0.0
    ndcg = dcg_at_k(hits) / dcg_at_k(sorted(hits, reverse=True)) if n_hits > 0 else 0.0
    hit_rate = 1.0 if n_hits > 0 else 0.0

    metrics_buffer["precision"].append(precision)
    metrics_buffer["recall"].append(recall)
    metrics_buffer["ndcg"].append(ndcg)
    metrics_buffer["hit_rate"].append(hit_rate)


def compute_ctr(now: int):
    """Compute CTR = % of recs that triggered watch within 30 min."""
    total, clicked = 0, 0
    for rec_time, user_id, recs in list(recommendations):
        if now - rec_time > CTR_LOOKAHEAD:
            continue
        w = [m for t, m in user_watched[user_id] if 0 <= t - rec_time <= CTR_LOOKAHEAD]
        if w and set(recs) & set(w):
            clicked += 1
        total += 1
    ctr = clicked / total if total > 0 else 0.0
    metrics_buffer["ctr"].append(ctr)
    return ctr, total, clicked


def compute_diversity_coverage():
    """Estimate genre and catalog coverage."""
    # Simple heuristic diversity proxy = unique movies recommended ratio
    all_recs = [m for _, _, recs in recommendations for m in recs]
    unique_movies = len(set(all_recs))
    coverage = unique_movies / max(1, len(all_recs))
    diversity = unique_movies / 1000.0  # assume ~1000 movies total
    metrics_buffer["diversity"].append(diversity)
    metrics_buffer["coverage"].append(coverage)
    return diversity, coverage


def aggregate_metrics():
    """Compute rolling averages of all metrics."""
    avg = lambda key: round(float(np.mean(metrics_buffer[key])), 4) if metrics_buffer[key] else 0.0
    avg_rating = (rating_sum / rating_count) if rating_count > 0 else 0.0
    avg_infer = round(float(np.mean(metrics_buffer["inference_time"])), 2) if metrics_buffer["inference_time"] else 0.0
    std_infer = round(float(np.std(metrics_buffer["inference_time"])), 2) if metrics_buffer["inference_time"] else 0.0
    return {
        "precision@K": avg("precision"),
        "recall@K": avg("recall"),
        "ndcg@K": avg("ndcg"),
        "hit_rate@K": avg("hit_rate"),
        "ctr": avg("ctr"),
        "diversity": avg("diversity"),
        "coverage": avg("coverage"),
        "avg_rating": round(avg_rating, 3),
        "avg_inference_time_ms": avg_infer,
        "std_inference_time_ms": std_infer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "users_tracked": len(user_watched),
    }


def stream_and_analyze():
    """Consume Kafka messages and compute rolling metrics."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    conf = {
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": GROUP_ID,
        "auto.offset.reset": "latest"
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])

    print(f"Connected to Kafka topic '{TOPIC}' on {KAFKA_BROKER}")
    print("Starting real-time online evaluation...")
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
                ctr, total, clicks = compute_ctr(now)
                diversity, coverage = compute_diversity_coverage()
                metrics = aggregate_metrics()
                os.system("clear")
                print("=" * 70)
                print("Real-Time Online Evaluation — Full Metric Dashboard")
                print("=" * 70)
                print(json.dumps(metrics, indent=2))
                print(f"Evaluated {total} recommendations ({clicks} clicks)")
                print(f"Users tracked: {metrics['users_tracked']}\n")

                # Save rolling snapshot for dashboard/report
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(metrics, f, indent=2)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        consumer.close()
        print("Kafka connection closed.")



if __name__ == "__main__":
    stream_and_analyze()
