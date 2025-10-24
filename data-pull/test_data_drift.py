#!/usr/bin/env python3
"""
Test script for data drift detection by pulling Kafka data for a specified duration.
"""

import logging
import sys
import time
import threading
from pathlib import Path

import pandas as pd

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




def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_data_stats(rating_path, watch_agg_path, movies_path, users_path) -> dict:
    """Get statistics about a list of parquet files."""
    ratings_df = pd.read_parquet(rating_path) if rating_path.exists() else pd.DataFrame()
    watches_df = pd.read_parquet(watch_agg_path) if watch_agg_path.exists() else pd.DataFrame()
    movies_df = pd.read_parquet(movies_path) if movies_path.exists() else pd.DataFrame()
    users_df = pd.read_parquet(users_path) if users_path.exists() else pd.DataFrame()

    # Join watches with users and movies to get complete watch records
    merged_watches = watches_df.merge(users_df, on="user_id", how="left").merge(movies_df, on="movie_id", how="left")   
    # Join ratings with users and movies to get complete rating records
    merged_ratings = ratings_df.merge(users_df, on="user_id", how="left").merge(movies_df, on="movie_id", how="left")
    # Calculate stats
    # Calculate average number of minutes watched per user
    avg_minutes_per_user = merged_watches.groupby("user_id").count().mean() if not merged_watches.empty else 0
    # Calculate average rating per user
    avg_rating_per_user = merged_ratings.groupby("user_id")["rating"].mean().mean() if not merged_ratings.empty else 0
    # Calculate User demographics stats by gender and age group 
    users_df['age_group'] = (users_df['age'] // 10) * 10
    demographics_stats = users_df.groupby(['gender', 'age_group']).size().to_dict() if not users_df.empty else {}
    return {
        "ratings": {
            "exists": rating_path.exists(),
            "rows": len(ratings_df),
            "avg_rating_per_user": avg_rating_per_user,
        },
        "watches": {
            "exists": watch_agg_path.exists(),
            "rows": len(watches_df),
            "avg_minutes_per_user": avg_minutes_per_user,
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




def print_statistics(before_stats: dict, after_stats: dict) -> None:
    """Print statistics about data collected."""
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    
    for key in ["ratings", "watches", "movies", "users"]:
        before = before_stats[key]
        after = after_stats[key]
        
        if not after["exists"]:
            print(f"\n{key.upper()}: No data file created")
            continue
            
        rows_before = before.get("rows", 0)
        rows_after = after.get("rows", 0)
        new_rows = rows_after - rows_before
        
        print(f"\n{key.upper()}:")
        print(f"  Rows before: {rows_before:,}")
        print(f"  Rows after:  {rows_after:,}")
        print(f"  New rows:    {new_rows:,}")
        
        if "error" in after:
            print(f"  Error: {after['error']}")
    
    print("\n" + "="*60)


def run_data_collection(duration_minutes: float) -> None:
    """
    Run data collection for a specified duration.
    
    Args:
        duration_minutes: Number of minutes to collect data
    """
    configure_logging()
    
    # Get statistics before collection
    print(f"\nStarting data collection for {duration_minutes} minute(s)...")
    before_stats = get_data_stats(RATINGS_PATH, WATCH_AGG_PATH, MOVIES_PATH, USERS_PATH)

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
    
    # Set up timer to stop after duration
    def stop_after_duration():
        time.sleep(duration_minutes * 60)
        logging.info(f"Duration of {duration_minutes} minute(s) reached. Stopping...")
        ingestor.shutdown_flag.set()
    
    timer = threading.Timer(duration_minutes * 60, stop_after_duration)
    timer.start()
    
    # Run the ingestor
    start_time = time.time()
    try:
        logging.info("Starting Kafka consumer...")
        while not ingestor.shutdown_flag.is_set():
            messages = ingestor.consumer.consume(
                num_messages=config.kafka_batch_size,
                timeout=config.poll_timeout,
            )
            
            if not messages:
                ingestor._maybe_flush(force=False)
                continue
            
            for msg in messages:
                if msg is None:
                    continue
                if msg.error():
                    continue
                ingestor.buffer_messages.append(msg)
            
            ingestor._maybe_flush(force=False)
            
            # Print progress every 30 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0 and len(ingestor.buffer_messages) > 0:
                logging.info(
                    f"Progress: {elapsed/60:.1f}/{duration_minutes} min, "
                    f"buffered: {len(ingestor.buffer_messages)} messages"
                )
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        timer.cancel()
        logging.info("Flushing final buffers...")
        ingestor._maybe_flush(force=True)
        ingestor.consumer.close()
    
    # Get statistics after collection
    after_stats = get_data_stats(RATINGS_PATH, WATCH_AGG_PATH, MOVIES_PATH, USERS_PATH)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nData collection completed in {elapsed_time/60:.2f} minutes")
    print_statistics(before_stats, after_stats)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test data collection from Kafka for data drift detection"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Duration in minutes to collect data (default: 5.0)",
    )
    
    args = parser.parse_args()
    
    if args.duration <= 0:
        print("Error: Duration must be greater than 0")
        sys.exit(1)
    
    run_data_collection(args.duration)


if __name__ == "__main__":
    main()
