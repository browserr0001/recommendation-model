#!/bin/bash

# Add cron job to retrain every 2 days at 2 AM
CRON_JOB="0 2 */2 * * /bin/bash retrain_cron.sh"

# Check if job already exists
if crontab -l 2>/dev/null | grep -q "retrain_cron.sh"; then
    echo "Cron job already exists"
else
    # Add the cron job
    (crontab -l 2>/dev/null; echo "${CRON_JOB}") | crontab -
    echo "Cron job added successfully"
fi

# Make the retrain script executable
chmod +x retrain_cron.sh

echo "Setup complete. Cron will run every 2 days at 2 AM"