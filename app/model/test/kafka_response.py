import re
import pandas as pd
from confluent_kafka import Consumer
import time
import numpy as np
import os

def parse_response_time(line):
    """Extract response time from a recommendation request log line"""
    try:
        # First check if this is a recommendation request
        if "recommendation request" not in line:
            return None
            
        # Extract the response time (e.g., "142 ms")
        ms_match = re.search(r'(\d+)\s*ms$', line)
        if ms_match:
            return int(ms_match.group(1))
        return None
    except Exception:
        return None

def stream_and_analyze():
    # Kafka consumer config
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'response-time-analyzer',
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
            response_time = parse_response_time(line)
            
            if response_time is not None:
                response_times.append(response_time)
                
            # Print stats periodically
            current_time = time.time()
            if current_time - last_print_time >= print_interval and response_times:
                last_print_time = current_time
                # Clear the terminal for better visibility (works on Mac/Linux)
                os.system('clear')

                print("="*40)
                print("Kafka Recommendation Response Time Monitor")
                print("="*40)
                
                avg_time = np.mean(response_times)
                median_time = np.median(response_times)
                min_time = np.min(response_times)
                max_time = np.max(response_times)
                p95_time = np.percentile(response_times, 95)
                # 600ms percentile
                p600ms = sum(1 for t in response_times if t > 600) / len(response_times) * 100

                
                print("\n--- Response Time Statistics ---")
                print(f"Total requests analyzed: {len(response_times)}")
                print(f"Average response time: {avg_time:.2f} ms")
                print(f"Median response time: {median_time:.2f} ms")
                print(f"Min response time: {min_time} ms")
                print(f"Max response time: {max_time} ms")
                print(f"600ms percentile: {p600ms:.2f}")
                print(f"Requests > 600ms: {sum(1 for t in response_times if t > 600)}")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        consumer.close()
        
        if response_times:
            # Final statistics
            print("\n--- Final Response Time Statistics ---")
            df = pd.DataFrame({'response_time': response_times})
            print(df['response_time'].describe())
            
            # # Save results
            # df.to_csv('response_times.csv', index=False)
            # print("Results saved to response_times.csv")

            # # output the responses with time > 600ms
            # slow_responses = df[df['response_time'] > 600]
            # slow_responses.to_csv('slow_responses.csv', index=False)
            # print("Slow responses (>600ms) saved to slow_responses.csv")

if __name__ == "__main__":
    stream_and_analyze()