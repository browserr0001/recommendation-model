#!/usr/bin/env python3
"""
Test script for data drift detection by pulling Kafka data for a specified duration.
"""

import logging
import sys
import time
import threading
from pathlib import Path
import os
import json
import pandas as pd
from confluent_kafka import KafkaError
# Import from data_pull.py
import data_pull
from data_pull import (
    Config,
    DataIngestor,
)

# Override the paths to use a different directory for testing
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

# Create subdirectories
(TEST_DATA_DIR / "ratings").mkdir(exist_ok=True)
(TEST_DATA_DIR / "watches").mkdir(exist_ok=True)
(TEST_DATA_DIR / "meta").mkdir(exist_ok=True)

# Override the global path variables in data_pull module
data_pull.RATINGS_PATH = TEST_DATA_DIR / "ratings" / "ratings.parquet"
data_pull.WATCH_AGG_PATH = TEST_DATA_DIR / "watches" / "watches.parquet"
data_pull.MOVIES_PATH = TEST_DATA_DIR / "meta" / "movies.parquet"
data_pull.USERS_PATH = TEST_DATA_DIR / "meta" / "users.parquet"

# Use the overridden paths
RATINGS_PATH = data_pull.RATINGS_PATH
WATCH_AGG_PATH = data_pull.WATCH_AGG_PATH
MOVIES_PATH = data_pull.MOVIES_PATH
USERS_PATH = data_pull.USERS_PATH

STATS_PATH = 'data_drift.json'


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_data_stats(rating_path, watch_agg_path, movies_path, users_path) -> dict:
    """Get statistics about a list of parquet files."""
    # check if files exist and read them
    if not rating_path.exists() or not watch_agg_path.exists() or not movies_path.exists() or not users_path.exists():
        return {
            "ratings": {"exists": rating_path.exists(), "rows": 0},
            "watches": {"exists": watch_agg_path.exists(), "rows": 0},
            "movies": {"exists": movies_path.exists(), "rows": 0},
            "users": {"exists": users_path.exists(), "rows": 0},
        }
    ratings_df = pd.read_parquet(rating_path) if rating_path.exists() else pd.DataFrame()
    watches_df = pd.read_parquet(watch_agg_path) if watch_agg_path.exists() else pd.DataFrame()
    movies_df = pd.read_parquet(movies_path) if movies_path.exists() else pd.DataFrame()
    users_df = pd.read_parquet(users_path) if users_path.exists() else pd.DataFrame()

    # Calculate stats
    # Calculate average number of minutes watched per user
    avg_minutes_per_user = watches_df.groupby("user_id")['minutes_watched'].sum().mean() if not watches_df.empty else 0
    # Calculate average rating per user
    avg_rating_per_user = ratings_df.groupby("user_id")["rating"].mean().mean() if not ratings_df.empty else 0
    # Calculate User demographics stats by gender and age group 
    users_df['age_group'] = (users_df['age'] // 10) * 10
    demographics_stats = users_df.groupby(['gender', 'age_group']).size().to_dict() if not users_df.empty else {}
    demographics_stats = {f"{k[0]}_{k[1]}": v/len(users_df) for k, v in demographics_stats.items()}

    return {
        "ratings": {
            "exists": rating_path.exists(),
            "rows": len(ratings_df),
            "avg_rating_per_user": float(avg_rating_per_user),
        },
        "watches": {
            "exists": watch_agg_path.exists(),
            "rows": len(watches_df),
            "avg_minutes_per_user": float(avg_minutes_per_user),
        },  
        "movies": {
            "exists": movies_path.exists(),
            "rows": len(movies_df),
        },
        "users": {
            "exists": users_path.exists(),
            "rows": len(users_df),
            "demographics_stats": demographics_stats,
        },
    }


def save_statistics(stats:dict) -> None:
    print(json.dumps(stats, indent=4))

    # load the json file and append the new stats
    with open(STATS_PATH, "r") as f:
        stats_before = json.load(f)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    stats_before[timestamp] = stats
    with open(STATS_PATH, "w") as f:
        json.dump(stats_before, f, indent=4)


def run_data_collection(duration_minutes: float) -> None:
    """
    Run data collection for a specified duration.
    
    Args:
        duration_minutes: Number of minutes to collect data
    """
    configure_logging()
    
    # Get statistics before collection
    print(f"\nStarting data collection for {duration_minutes} minute(s)...")

    # clear data_drift.json if exists
    if os.path.exists(STATS_PATH):
        os.remove(STATS_PATH)

    # create the file again
    with open(STATS_PATH, "w") as f:
        json.dump({}, f)

    # Clear the data for fresh statistics
    for path in [RATINGS_PATH, WATCH_AGG_PATH, MOVIES_PATH, USERS_PATH]:
        if path.exists():
            path.unlink()
    
    # Create config with appropriate flush settings for testing
    config = Config(
        bootstrap_servers="localhost:9092",
        topic="movielog8",
        group_id=f"test-data-drift",
        poll_timeout=1.0,
        kafka_batch_size=1000,
        flush_message_count=5000,  # Flush more frequently for testing
        flush_interval_seconds=10.0,  # Flush every 10 seconds
        base_metadata_url="http://128.2.220.241:8080",
        metadata_workers=8,
        request_timeout=10.0,
    )
    
    # Create ingestor
    ingestor = DataIngestor(config)
    
    # Run the ingestor
    start_time = time.time()
    last_stats_time = start_time
    interval_seconds = duration_minutes * 60
    
    try:
        logging.info("Starting data collection...")
        
        while not ingestor.shutdown_flag.is_set():
            messages = ingestor.consumer.consume(
                num_messages=ingestor.config.kafka_batch_size,
                timeout=ingestor.config.poll_timeout,
            )
            if not messages:
                ingestor._maybe_flush(force=False)
            else:
                for msg in messages:
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        logging.error("Kafka error: %s", msg.error())
                        continue
                    ingestor.buffer_messages.append(msg)
                ingestor._maybe_flush(force=False)
            
            # Check if it's time to output statistics
            current_time = time.time()
            elapsed_since_last_stats = current_time - last_stats_time
            
            if elapsed_since_last_stats >= interval_seconds:
                # Flush any pending data before collecting stats
                ingestor._maybe_flush(force=True)
                
                # Get statistics after collecting for duration_minutes
                after_stats = get_data_stats(RATINGS_PATH, WATCH_AGG_PATH, MOVIES_PATH, USERS_PATH)
                after_stats["collection_duration_minutes"] = duration_minutes

                # Print and save summary
                logging.info(f"Outputting statistics after {duration_minutes} minute(s) of collection")
                save_statistics(after_stats)
                
                # Clear the data for fresh statistics for the next interval
                for path in [RATINGS_PATH, WATCH_AGG_PATH, MOVIES_PATH, USERS_PATH]:
                    if path.exists():
                        path.unlink()
                
                # Reset the timer for the next interval
                last_stats_time = current_time
    except KeyboardInterrupt:
        logging.info("Interrupt received; flushing buffers before exit.")
        ingestor.shutdown_flag.set()
        ingestor._maybe_flush(force=True)
    finally:
        ingestor.consumer.close()
        logging.info("Consumer closed. Shutdown complete.")
        time.sleep(2)  # Wait for any final flushes


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test data collection from Kafka for data drift detection"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duration in minutes to collect data (default: 5.0)",
    )
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        print("Error: Duration must be greater than 0")
        sys.exit(1)
    
    run_data_collection(args.duration)


if __name__ == "__main__":
    main()
