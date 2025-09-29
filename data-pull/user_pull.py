#!/usr/bin/env python3
import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_URL = "http://128.2.220.241:8080"
OUTPUT_FILE = "data/meta/users_new.parquet"
LOG_FILE = "data/meta/users_log.txt"
START_UID = 1
END_UID = 1_100_000
BATCH_SIZE = 1000      # save every N users
WORKERS = 50           # parallel threads
TIMEOUT = 5            # seconds per request
# ----------------------------------------

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- Load existing parquet if present ---
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_parquet(OUTPUT_FILE)
    fetched_ids = set(existing["user_id"].tolist())
    print(f"Resuming: already have {len(fetched_ids)} users")
else:
    existing = pd.DataFrame()
    fetched_ids = set()
    print("Starting fresh")

# --- Function to fetch a single user ---
def fetch_user(uid):
    if uid in fetched_ids:
        return None
    url = f"{BASE_URL}/user/{uid}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            data["user_id"] = int(data.get("user_id", uid))
            return data
    except Exception:
        pass
    return None

# --- Loop over ranges of IDs ---
buffer = []
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(fetch_user, uid): uid for uid in range(START_UID, END_UID + 1)}

    for fut in tqdm(as_completed(futures), total=len(futures)):
        data = fut.result()
        if data:
            buffer.append(data)

        # Write batch when full
        if len(buffer) >= BATCH_SIZE:
            df_new = pd.DataFrame(buffer)
            if not existing.empty:
                existing = pd.concat([existing, df_new], ignore_index=True)
            else:
                existing = df_new
            existing.to_parquet(OUTPUT_FILE, index=False)

            # log user_ids
            with open(LOG_FILE, "a") as f:
                for row in buffer:
                    f.write(f"{row['user_id']}\n")

            print(f"Saved batch of {len(buffer)} users. Total so far: {len(existing)}")
            buffer = []

# --- Save any leftovers ---
if buffer:
    df_new = pd.DataFrame(buffer)
    if not existing.empty:
        existing = pd.concat([existing, df_new], ignore_index=True)
    else:
        existing = df_new
    existing.to_parquet(OUTPUT_FILE, index=False)
    with open(LOG_FILE, "a") as f:
        for row in buffer:
            f.write(f"{row['user_id']}\n")
    print(f"Final save: {len(buffer)} users. Total so far: {len(existing)}")