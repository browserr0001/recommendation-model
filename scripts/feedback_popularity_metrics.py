import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "app" / "logs"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only last week of files
LAST_N_DAYS = 7

# Read only the first N lines of each daily log
HEAD_LINES = 50000   # tune: 10000 / 50000 / 100000
START_LINE = 0


def compute_day_metrics(path: str):
    movie_counts = Counter()
    num_reqs = 0
    total_slots = 0
    bad_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < START_LINE:
                continue
            if i >= START_LINE + HEAD_LINES:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            recs = obj.get("recommendations", [])
            if not isinstance(recs, list) or not recs:
                continue

            num_reqs += 1
            total_slots += len(recs)

            for mid in recs:
                movie_counts[mid] += 1

    if total_slots == 0:
        print(f"[WARN] No valid recs in head of {path}. bad_lines={bad_lines}")
        return None

    unique_movies = len(movie_counts)

    top_10_share = sum(c for _, c in movie_counts.most_common(10)) / total_slots
    top_50_share = sum(c for _, c in movie_counts.most_common(50)) / total_slots

    head_n = max(1, int(0.01 * unique_movies))
    top_1pct_share = sum(c for _, c in movie_counts.most_common(head_n)) / total_slots

    fname = Path(path).name
    date = (
        fname.replace("predictions_", "")
             .replace(".jsonl", "")
             .replace(".json", "")
    )

    return {
        "date": date,
        "num_requests_in_head": num_reqs,
        "total_slots_in_head": total_slots,
        "unique_movies_recommended": unique_movies,
        "top_10_share": top_10_share,
        "top_50_share": top_50_share,
        "top_1pct_share": top_1pct_share,
        "start_line": START_LINE,
        "head_lines": HEAD_LINES,
    }

def main():
    print("LOGS_DIR:", LOGS_DIR)
    print("LOGS_DIR exists?", LOGS_DIR.exists())
    if not LOGS_DIR.exists():
        print("[ERROR] LOGS_DIR not found.")
        return

    files = sorted(LOGS_DIR.glob("predictions_*.jsonl"))
    files = [str(p) for p in files][-LAST_N_DAYS:]

    print("Matched prediction files (last week):", files)
    print("Using HEAD_LINES =", HEAD_LINES)

    results = []
    for fp in files:
        m = compute_day_metrics(fp)
        if m:
            results.append(m)

    out_json = OUT_DIR / "feedback_popularity_daily.json"
    out_csv = OUT_DIR / "feedback_popularity_daily.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    else:
        print("[WARN] results empty.")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")

if __name__ == "__main__":
    main()
