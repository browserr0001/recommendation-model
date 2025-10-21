import re
from collections import defaultdict, deque
from confluent_kafka import Consumer
import time
import numpy as np
import os 
import sys

# Parameters
WINDOW_SIZE = 1000  # Number of users to keep for rolling evaluation

# Data structures to track user interactions
user_watched = defaultdict(set)  # user_id -> set of watched movie_ids
user_rated = defaultdict(set)    # user_id -> set of rated movie_ids
user_recs = deque(maxlen=WINDOW_SIZE)  # (user_id, recommended_movies, timestamp)

# Regex patterns
rec_pattern = re.compile(r'^\S+?,(\d+),recommendation request.*?result: (.+?), \d+ ms')
watch_pattern = re.compile(r'^(.*?),(\d+),GET /data/m/(.+?)/\d+\.mpg')
rating_pattern = re.compile(r'^(.*?),(\d+),GET /rate/(.+)=(\d+)')

# Online evaluation metrics
hit_rates = []

# Function to parse a line and update state
def process_line(line):
    # Recommendation event
    rec_match = rec_pattern.match(line)
    if rec_match:
        ts = rec_match.group(0)
        user_id = int(rec_match.group(1))
        recs = [m.strip() for m in rec_match.group(2).split(',')]
        user_recs.append((user_id, recs, ts))
        # Evaluate hit rate if we have watch/rating info
        watched = user_watched[user_id]
        rated = user_rated[user_id]
        relevant = watched | rated
        if relevant:
            hits = sum(1 for m in recs if m in relevant)
            hit_rate = hits / len(recs)
            hit_rates.append(hit_rate)
            print(f"User {user_id} | Hit Rate: {hit_rate:.2f} | {hits}/{len(recs)} | Time: {ts}")
        return
    # Watch event
    watch_match = watch_pattern.match(line)
    if watch_match:
        ts, user_id, movie_id = watch_match.groups()
        user_id = int(user_id)
        user_watched[user_id].add(movie_id)
        return
    # Rating event
    rating_match = rating_pattern.match(line)
    if rating_match:
        ts, user_id, movie_id, rating = rating_match.groups()
        user_id = int(user_id)
        user_rated[user_id].add(movie_id)
        return

def stream_and_analyze():
    # Kafka consumer config
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'online-eval', 
        'auto.offset.reset': 'latest'
    }
    
    consumer = Consumer(conf)
    consumer.subscribe(['movielog8'])
    
    # Track response times
    response_times = []
    last_print_time = time.time()
    print_interval = 5  # Print stats every 5 seconds
    
    try:
        print("Starting Kafka stream analysis... (Press Ctrl+C to stop)")
        print("Waiting for recommendation requests...")
        
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
                
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
                
            line = msg.value().decode('utf-8')
            process_line(line)
            
            if hit_rates:
                current_time = time.time()

                if current_time - last_print_time >= print_interval and response_times:
                    last_print_time = current_time
                    # Clear the terminal for better visibility (works on Mac/Linux)
                    os.system('clear')

                    print("="*40)
                    print("Kafka Recommendation Response Time Monitor")
                    print("="*40)
                    avg_hit = sum(hit_rates) / len(hit_rates)
                    print(f"\n--- Online Evaluation Summary ---")
                    print(f"Evaluated {len(hit_rates)} recommendations.")
                    print(f"Average Hit Rate: {avg_hit:.3f}")
            # else:
            #     print("No recommendations evaluated yet.")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        consumer.close()
        
        if response_times:
            # Final statistics
            print("\n--- Final Response Time Statistics ---")

def process_from_stdin():
    print("Reading from stdin (pipe kcat output here)...")
    try:
        for line in sys.stdin:
            process_line(line)
            
            # Show summary periodically
            if len(hit_rates) % 10 == 0 and hit_rates:
                avg_hit = sum(hit_rates) / len(hit_rates)
                print(f"\n--- Online Evaluation Summary ---")
                print(f"Evaluated {len(hit_rates)} recommendations.")
                print(f"Average Hit Rate: {avg_hit:.3f}")
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == '--kafka':
    #     stream_and_analyze()
    # else:
    #     process_from_stdin()
    stream_and_analyze()