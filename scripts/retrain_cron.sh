#!/bin/bash
# filepath: scripts/retrain_cron.sh

set -e  # Exit on error

# Configuration
LOG_DIR="logs/retraining"
LOG_FILE="$LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="/tmp/model_retrain.lock"

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect all output to log file
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "==================================="
echo "Model Retraining Job Started"
echo "Date: $(date)"
echo "==================================="

# Check for lock file to prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    echo "ERROR: Another retraining job is running (lock file exists)"
    exit 1
fi

# Create lock file
touch "$LOCK_FILE"

# Cleanup function to remove lock file on exit
cleanup() {
    rm -f "$LOCK_FILE"
    echo "Lock file removed"
}
trap cleanup EXIT

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# 1. Pull latest data from DVC remote (if configured)
echo "Step 1: Syncing data with DVC remote..."
dvc pull data-pull/data/ || echo "Warning: DVC pull failed or no remote configured"

# 2. Add/update data files in DVC
echo "Step 2: Adding data files to DVC..."
dvc add data-pull/data/ratings/ratings.parquet
dvc add data-pull/data/watches/watches.parquet
dvc add data-pull/data/meta/movies.parquet
dvc add data-pull/data/meta/users_new.parquet

# 3. Commit data changes to git
echo "Step 3: Committing data version to git..."
git add data-pull/data/ratings/ratings.parquet.dvc \
        data-pull/data/watches/watches.parquet.dvc \
        data-pull/data/meta/movies.parquet.dvc \
        data-pull/data/meta/users_new.parquet.dvc

if git diff --cached --quiet; then
    echo "No data changes detected"
else
    git commit -m "Update data version $(date +%Y-%m-%d)"
fi

# 4. Update params.yaml to mark this as production training
echo "Step 4: Configuring training parameters..."
python -c "
import yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
params['train']['train_for_prod'] = True
params['train']['tag'] = 'Updated Model'
with open('params.yaml', 'w') as f:
    yaml.dump(params, f)
"

# 5. Run DVC pipeline to retrain model
echo "Step 5: Running DVC pipeline to retrain model..."
dvc repro train

# 6. Push DVC outputs (model) to remote
echo "Step 6: Pushing model to DVC remote..."
dvc push app/pkl_models/content_based_model_full.pkl || echo "Warning: DVC push failed or no remote configured"

# 7. Commit pipeline outputs
echo "Step 7: Committing pipeline changes..."
git add dvc.lock app/pkl_models/content_based_model_full.pkl.dvc params.yaml
git commit -m "Retrain model $(date +%Y-%m-%d) - Updated Model tag" || echo "No changes to commit"

# 8. Verify model is in production
echo "Step 8: Verifying model deployment..."
python -c "
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = 'content_based_model_full'

# Get production models
prod_versions = client.get_latest_versions(model_name, stages=['Production'])
if prod_versions:
    latest = prod_versions[0]
    tags = client.get_model_version(model_name, latest.version).tags
    print(f'✓ Model version {latest.version} is in Production')
    print(f'  Tags: {tags}')
    if tags.get('model_tag') == 'Updated Model':
        print('✓ Model has correct tag: Updated Model')
    else:
        print('✗ Warning: Model tag mismatch')
else:
    print('✗ No production model found!')
    exit(1)
"

# 9. Push to git remote
echo "Step 9: Pushing to git remote..."
git push origin main || echo "Warning: Git push failed"

# 10. Log success metrics
echo "Step 10: Logging completion metrics..."
python -c "
import json
from datetime import datetime
import os

log_entry = {
    'timestamp': datetime.now().isoformat(),
    'status': 'success',
    'log_file': '$LOG_FILE'
}

log_path = 'logs/retraining/history.jsonl'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
"

echo "==================================="
echo "Model Retraining Job Completed Successfully"
echo "Date: $(date)"
echo "==================================="