#!/bin/bash
set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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
echo "Working Directory: $PROJECT_ROOT"
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

# Set up Git credentials for cron (if needed)
# Uncomment and configure if running from cron
# export GIT_AUTHOR_NAME="Automated Retraining"
# export GIT_AUTHOR_EMAIL="retrain@example.com"
# export GIT_COMMITTER_NAME="Automated Retraining"
# export GIT_COMMITTER_EMAIL="retrain@example.com"


# Update params.yaml to mark this as production training
echo "Step: Configuring training parameters..."
python3 -c "
import yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)
params['train']['train_for_prod'] = True
params['train']['tag'] = 'Updated_Model'
with open('params.yaml', 'w') as f:
    yaml.dump(params, f, default_flow_style=False)
print('✓ Updated params.yaml with tag: Updated_Model')
"

# Run DVC pipeline to retrain ONLY the updated model
echo "Step: Running DVC pipeline to retrain Updated_Model..."
dvc repro train_updated_model


# 1. Pull latest data from DVC remote (if configured)
# echo "Step 1: Syncing data with DVC remote..."
# dvc pull data-pull/data/ || echo "Warning: DVC pull failed or no remote configured"

# 2. Add/update data files in DVC
echo "Step: Adding data files to DVC..."
dvc add data-pull/data/ratings/ratings.parquet
dvc add data-pull/data/watches/watches.parquet
dvc add data-pull/data/meta/movies.parquet
dvc add data-pull/data/meta/users_new.parquet
dvc add data-pull/data/meta/users.parquet

# Push to dvc remote
echo "Step: Pushing data files to DVC Remote..."
dvc push data-pull/data/meta/movies.parquet
dvc push data-pull/data/meta/users.parquet
dvc push data-pull/data/ratings/ratings.parquet
dvc push data-pull/data/watches/watches.parquet

# Commit data changes to git
echo "Step: Committing data version to git..."
git add data-pull/data/ratings/ratings.parquet.dvc \
        data-pull/data/watches/watches.parquet.dvc \
        data-pull/data/meta/movies.parquet.dvc \
        data-pull/data/meta/users_new.parquet.dvc \
        data-pull/data/meta/users.parquet.dvc

if git diff --cached --quiet; then
    echo "No data changes detected"
else
    git commit -m "Update data version $(date +%Y-%m-%d)" || echo "No changes to commit"
fi

# Push DVC outputs (model) to remote
echo "Step: Pushing model to DVC remote..."
dvc push app/pkl_models/content_based_model_Updated_Model.pkl || echo "Warning: DVC push failed or no remote configured"

# Commit pipeline outputs and model version
echo "Step: Committing pipeline changes..."
git add dvc.lock params.yaml

if git diff --cached --quiet; then
    echo "No pipeline changes to commit"
else
    git commit -m "Retrain Updated_Model $(date +%Y-%m-%d) [automated]" || echo "Commit failed or no changes"
fi

# Verify model file exists and get metadata
echo "Step: Verifying model deployment..."
python3 -c "
import pickle
import os
from pathlib import Path

model_path = 'app/pkl_models/content_based_model_Updated_Model.pkl'

if not os.path.exists(model_path):
    print('✗ Model file not found!')
    exit(1)

try:
    # Try to load and verify model
    import sys
    sys.path.insert(0, 'app/model/src')
    
    # Use custom unpickler to handle module path issues
    class ForceImportUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'ContentBasedRecommender':
                from content_based import ContentBasedRecommender
                return ContentBasedRecommender
            return super().find_class(module, name)
    
    with open(model_path, 'rb') as f:
        model = ForceImportUnpickler(f).load()
    
    # Get model metadata
    metadata = model.get_metadata()
    print(f'✓ Model successfully loaded and verified')
    print(f'  Model Tag: {metadata.get(\"model_tag\", \"Unknown\")}')
    print(f'  Trained At: {metadata.get(\"trained_at\", \"Unknown\")}')
    print(f'  Git Commit: {metadata.get(\"git_commit_hash\", \"Unknown\")[:8]}')
    print(f'  Data Version: {metadata.get(\"data_version\", \"Unknown\")[:50]}...')
    
    # Verify it has correct tag
    if metadata.get('model_tag') == 'Updated_Model':
        print('✓ Model has correct tag: Updated_Model')
    else:
        print(f'✗ Warning: Model tag mismatch: {metadata.get(\"model_tag\")}')
        exit(1)
        
except Exception as e:
    print(f'✗ Error loading model: {e}')
    exit(1)

print(f'✓ Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB')
"

# Push to git remote
echo "Step: Pushing to git remote..."
git push || echo "Warning: Git push failed (check remote configuration)"

# Log success metrics
echo "Step: Logging completion metrics..."
python3 -c "
import json
from datetime import datetime
import os

log_entry = {
    'timestamp': datetime.now().isoformat(),
    'status': 'success',
    'model_tag': 'Updated_Model',
    'log_file': '$LOG_FILE',
    'dvc_stage': 'train_updated_model'
}

log_path = 'logs/retraining/history.jsonl'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'a') as f:
    f.write(json.dumps(log_entry) + '\n')

print('✓ Retraining history logged')
"

echo "==================================="
echo "Model Retraining Job Completed Successfully"
echo "Model: content_based_model_Updated_Model.pkl"
echo "Date: $(date)"
echo "==================================="