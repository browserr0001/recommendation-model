import os
import pickle
import sys
import subprocess

import yaml
from content_based import ContentBasedRecommender
import pyarrow.parquet as pq
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

import json

import mlflow.pyfunc


# Set up MLflow experiment
experiment_name = "content_based_recommender"
mlflow.set_experiment(experiment_name)

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
    watches_hash = watches_yaml['outs'][0]['hash']

    ratings_path = os.path.join(path, 'ratings/ratings.parquet.dvc')
    with open(ratings_path, 'r') as f:
        ratings_yaml = yaml.safe_load(f)
    ratings_hash = ratings_yaml['outs'][0]['hash']

    users_path = os.path.join(path, 'meta/users_new.parquet.dvc')
    with open(users_path, 'r') as f:
        users_yaml = yaml.safe_load(f)
    users_hash = users_yaml['outs'][0]['hash']

    movies_path = os.path.join(path, 'meta/movies.parquet.dvc')
    with open(movies_path, 'r') as f:
        movies_yaml = yaml.safe_load(f)
    movies_hash = movies_yaml['outs'][0]['hash']

    return movies_hash, users_hash, ratings_hash, watches_hash

class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.recommender: ContentBasedRecommender = None
        self.metadata = {}

    def load_context(self, context):
        from joblib import load
        self.recommender = load(context.artifacts["model_pkl"])
        self.metadata = self.recommender.get_metadata()

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, int):
            return self.recommender.get_recommendations(user_id=model_input)
        elif isinstance(model_input, list):
            return [self.recommender.get_recommendations(uid) for uid in model_input]
        elif isinstance(model_input, pd.DataFrame):
            results = []
            for _, row in model_input.iterrows():
                uid = int(row["user_id"])
                n = int(row.get("top_n", 20))
                recs = self.recommender.get_recommendations(user_id=uid, top_n=n)
                results.append(recs)
            return results
        return self.recommender.get_recommendations(user_id=model_input)

    def get_metadata(self):
        return self.metadata

def train(movies, users, ratings, watches, mid_rating_watch_over: float, age_bucket: int, metadata: dict = None):
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
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies, users, ratings, watches, mid_rating_watch_over=mid_rating_watch_over, age_bucket=age_bucket, metadata=metadata)
    return content_recommender

def read_data(path_prefix='data/'):
    # check this folder contains necessary parquet files
    if not (os.path.exists(f'{path_prefix}meta/movies.parquet')) or \
            not (os.path.exists(f'{path_prefix}meta/users_new.parquet')) or \
            not (os.path.exists(f'{path_prefix}ratings/ratings.parquet')) or \
            not (os.path.exists(f'{path_prefix}watches/watches.parquet')):
        raise FileNotFoundError("One or more required parquet files are missing in the specified path.")

    movies = pq.read_table(f'{path_prefix}meta/movies.parquet').to_pandas()
    users = pq.read_table(f'{path_prefix}meta/users_new.parquet').to_pandas()
    ratings = pq.read_table(f'{path_prefix}ratings/ratings.parquet').to_pandas()
    watches = pq.read_table(f'{path_prefix}watches/watches.parquet').to_pandas()

    selected_movie_cols = ['id', 'title', 'adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
    movies = movies[selected_movie_cols]
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['release_year'] = movies['release_date'].dt.year
    movies.rename(columns={'id': 'movie_id'}, inplace=True)
    

    def get_names(names):
        return [g['name'] for g in names]
    movies['genres'] = movies['genres'].apply(get_names)
    movies['production_companies'] = movies['production_companies'].apply(get_names)
    movies['production_countries'] = movies['production_countries'].apply(get_names)

    numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                           'vote_average', 'vote_count']
    for feature in numeric_features:
        movies[feature] = pd.to_numeric(movies[feature], errors='coerce')

    all_features = ['adult', 'budget', 'genres', 'original_language', 'overview',
                    'popularity', 'production_companies', 'production_countries', 
                    'release_date', 'release_year'] + numeric_features 
    assert all(feature in movies.columns for feature in all_features)
    # convert adult to boolean
    movies['adult'] = movies['adult'].astype(bool)
    return movies, users, ratings, watches

def main():
    params = yaml.safe_load(open("params.yaml"))["train"]
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py output\n")
        sys.exit(1)

    output = sys.argv[1]
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

   

    # Log the model with MLflow
    with mlflow.start_run() as run:
        # Prepare metadata for the model
        metadata = {
            'mlflow_run_id': run.info.run_id,
            'git_commit_hash': git_commit_hash,
            'git_branch': git_branch,
            'pipeline_version': params.get("pipeline_version", "Unknown"),
            'data_version': data_hash,
        }
        
        # Log parameters first
        mlflow.log_param("mid_rating_watch_over", mid_rating_watch_over)
        mlflow.log_param("age_bucket", age_bucket)
        mlflow.log_param("model_type", "content_based")
        mlflow.log_param("pipeline_version", params.get("pipeline_version", "Unknown"))
        mlflow.log_param("data_version", data_hash)
        mlflow.log_param("git_commit_hash", git_commit_hash)
        mlflow.log_param("git_branch", git_branch)
        
        # Train model with metadata
        model = train(movies, users, ratings, watches, mid_rating_watch_over=mid_rating_watch_over, age_bucket=age_bucket, metadata=metadata)
        
        # Save the model locally
        with open(output, "wb") as fd:
            pickle.dump(model, fd)

        # Save metadata to JSON file in the location in the same folder with the model
        metadata_path = output.replace('.pkl', '_metadata.json')
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        artifacts = {
            "model_pkl": output, 
            "model_metadata": metadata_path
        }

        # Log the pickle file as an artifact
        model_name = "content_based_model_full"

        mlflow.pyfunc.log_model(
            name="recommender_model",
            python_model=RecommenderWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name, 
            input_example=pd.DataFrame({"user_id": [int(users['user_id'].iloc[0])]})
        )

        if params.get("train_for_prod", False):
            client = MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
            latest_version_number = latest_versions[0].version

            # Promote to Production and archive any existing production versions
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version_number,
                stage="Production",
                archive_existing_versions=True 
            )

            # Set tags on the model version
            model_tag = params.get("tag", "default")
            client.set_model_version_tag(
                name=model_name,
                version=latest_version_number,
                key="model_tag",
                value=model_tag
            )

            # Update model metadata with registry information
            model.metadata['registered_model_name'] = model_name
            model.metadata['model_tag'] = model_tag

            # Re-save the model with updated metadata
            with open(output, "wb") as fd:
                pickle.dump(model, fd)
            
            # Log the model version info as parameters
            mlflow.log_param("registered_model_name", model_name)
            mlflow.log_param("model_tag", model_tag)
        
        print(f"Model training complete. Run ID: {run.info.run_id}")
        print(f"Model saved to: {output}")
        print(f"Data version: {data_hash}")
        print(f"Code version: {git_commit_hash}")
    
    return model

if __name__ == "__main__":
    main()
