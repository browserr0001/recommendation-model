## Data Pull Script Setup

1. Open the SSH tunnel so the local machine can reach Kafka:

   ```
   ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.220.241 -NTf
   ```

2. Review the configuration block near the top of `data-pull/data_pull.py`. Update the constants if you need different Kafka settings, batch sizes, or metadata service details.

3. Install Python dependencies (inside your preferred environment):

   ```
   pip install confluent-kafka pandas pyarrow requests
   ```

4. Run the puller whenever you want fresh data:

   ```
   python data-pull/data_pull.py
   ```

5. Stop the script with `Ctrl+C` when you are done. Offsets are committed after each flush, so rerunning the script continues from the latest point.
