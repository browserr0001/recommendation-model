import os
import pickle
import sys
import subprocess
import tempfile
import shutil
import yaml
from content_based import ContentBasedRecommender
from content_based import read_data
import pyarrow.parquet as pq
import pandas as pd


def get_git_commit_hash():
    """Get current git commit hash for code versioning"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit_hash
    except Exception as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return "unknown"

def get_git_branch():
    """Get current git branch"""
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        return branch
    except Exception as e:
        print(f"Warning: Could not get git branch: {e}")
        return "unknown"

def get_dvc_hash(path):
    watches_path = os.path.join(path, 'watches/watches.parquet.dvc')
    with open(watches_path, 'r') as f:
        watches_yaml = yaml.safe_load(f)
    watches_hash = watches_yaml['outs'][0]['md5']

    ratings_path = os.path.join(path, 'ratings/ratings.parquet.dvc')
    with open(ratings_path, 'r') as f:
        ratings_yaml = yaml.safe_load(f)
    ratings_hash = ratings_yaml['outs'][0]['md5']

    users_path = os.path.join(path, 'meta/users_new.parquet.dvc')
    with open(users_path, 'r') as f:
        users_yaml = yaml.safe_load(f)
    users_hash = users_yaml['outs'][0]['md5']

    movies_path = os.path.join(path, 'meta/movies.parquet.dvc')
    with open(movies_path, 'r') as f:
        movies_yaml = yaml.safe_load(f)
    movies_hash = movies_yaml['outs'][0]['md5']

    return movies_hash, users_hash, ratings_hash, watches_hash

def train(params, output:str):
    """
    Train a content-based recommender.

    Args:
        movies (pd.DataFrame): Movies data.
        users (pd.DataFrame): Users data.
        ratings (pd.DataFrame): Ratings data.
        watches (pd.DataFrame): Watches data.
        metadata (dict): Metadata for provenance tracking

    Returns:
        model (ContentBasedRecommender): Trained model.
    """
    data_path = params["data_path"]
    age_bucket = params["age_bucket"]
    mid_rating_watch_over = params["mid_rating_watch_over"]

    # Get Git commit hash and branch for code versioning
    git_commit_hash = get_git_commit_hash()
    git_branch = get_git_branch()
    
    print(f"Training model with git commit: {git_commit_hash} (branch: {git_branch})")

    # Load the data
    movies, users, ratings, watches = read_data(data_path)
    
    # Get DVC hashes for data versioning
    movies_hash, users_hash, ratings_hash, watches_hash = get_dvc_hash("data-pull/data/")
    data_hash = f"movies:{movies_hash},users:{users_hash},ratings:{ratings_hash},watches:{watches_hash}"
    
    print(f"Training with data version: {data_hash}")
    metadata = {
            'model_name': "ContentBasedRecommender",
            'model_tag': params.get("model_tag", "Updated_Model"),
            'git_commit_hash': git_commit_hash,
            'git_branch': git_branch,
            'pipeline_version': params.get("pipeline_version", "Unknown"),
            'data_version': data_hash,
        }

    model = ContentBasedRecommender()
    model.fit(movies, users, ratings, watches, mid_rating_watch_over=mid_rating_watch_over, age_bucket=age_bucket, metadata=metadata, num_warm_users=params.get("num_warm_users", 10000))
    
    return model


def atomic_copy_file(source_path, destination_path):
    """
    Atomically copies a file, ensuring both the original and the new file
    exist and are consistent.

    Args:
        source_path (str): The path to the original file.
        destination_path (str): The path for the new, copied file.
    """
    # Create a temporary file in the same directory as the destination
    # This is crucial for os.rename/os.replace to be atomic.
    temp_dir = os.path.dirname(destination_path)
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
        temp_file_path = temp_file.name

    try:
        # Copy the content of the source file to the temporary file
        shutil.copy2(source_path, temp_file_path)

        # Atomically rename the temporary file to the destination path
        # os.replace is preferred for Python 3.3+ as it handles overwriting
        # and is atomic on more systems (including Windows).
        if hasattr(os, 'replace'):
            os.replace(temp_file_path, destination_path)
        else:
            os.rename(temp_file_path, destination_path)

    except Exception as e:
        # Clean up the temporary file if an error occurs
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e

def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py output\n")
        sys.exit(1)

    output = sys.argv[1]

    # Train the model 
    model = train(params, output)
    # Save with a versioned filename
    version_tag = params.get("tag", "Updated_Model")
    save_path = f"{output}/content_based_model_{version_tag}.pkl"
    train_time = model.get_metadata().get('trained_at', 'unknown')
    archive_path = f"archive/model_store/content_based_model_{version_tag}_{train_time}.pkl"
    print(f"Saving model to {save_path}...")
    # Save the model to the archive path first
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
    with open(archive_path, "wb") as f:
        pickle.dump(model, f)

    # copy the model to save_path with atomic write for the docker container to consume and avoid corruption
    atomic_copy_file(archive_path, save_path)
    print(f"Model saved to {save_path}")
    

if __name__ == "__main__":
    main()
