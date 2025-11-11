# import os
# import pickle
# from app.model.src.content_based import ContentBasedRecommender

# class ForceImportUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if name == 'ContentBasedRecommender':
#             # This always returns the right class, no matter the module path in pickle
#             return ContentBasedRecommender
#         return super().find_class(module, name)

# def get_model():
#     print("loading model...")
#     model_path = os.path.join(os.path.dirname(__file__), 'pkl_models', 'content_based_model_full.pkl')
#     with open(model_path, 'rb') as f:
#         model = ForceImportUnpickler(f).load()
#     return model


from mlflow import MlflowClient
import mlflow.pyfunc
import os, sys
import json

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "src"))
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "content_based_model_full"
STAGE = "Production"

mlflow.set_tracking_uri("file:/mlruns")
class ModelWithMetadata:
    def __init__(self, pyfunc_model, metadata: dict, model_version=None, run_id=None):
        self._model = pyfunc_model
        self._metadata = metadata or {}
        if model_version:
            self._metadata.setdefault("model_version", model_version)
        if run_id:
            self._metadata.setdefault("run_id", run_id)

    def predict(self, user_id, top_n=20):
        # Pass as dict to the pyfunc model so RecommenderWrapper can extract top_n
        model_input = {"user_id": user_id, "top_n": top_n}
        return self._model.predict(model_input)[0]

    def get_metadata(self):
        return self._metadata


def find_model_version_by_tag(client: MlflowClient, model_name: str, tag_key: str, tag_value: str):
    """Find model version by tag key and value"""
    try:
        # Search all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        for version in versions:
            # Check if this version has the matching tag
            if version.tags.get(tag_key) == tag_value:
                return version
        return None
    except Exception as e:
        print(f"Error searching for model by tag: {e}")
        return None

def get_model(tag: str = None):
    """
    Load model from MLflow registry.
    
    Args:
        tag: Optional tag value to load a specific model version.
             If None, loads the latest Production model.
             Tag is matched against the 'model_tag' key.
    """
    client = MlflowClient()
    model_version_obj = None
    
    # If tag is specified, search for model version with that tag
    if tag:
        print(f"Searching for model {MODEL_NAME} with tag '{tag}'...")
        model_version_obj = find_model_version_by_tag(client, MODEL_NAME, "model_tag", tag)
        if not model_version_obj:
            print(f"Warning: No model found with tag '{tag}', falling back to latest {STAGE} model")
    
    # Fall back to latest Production model if no tag or tag not found
    if not model_version_obj:
        print(f"Loading latest {STAGE} model for {MODEL_NAME}...")
        versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])
        model_version_obj = versions[0] if versions else None
    
    if not model_version_obj:
        raise ValueError(f"No model version found for {MODEL_NAME}")
    
    model_version = model_version_obj.version
    run_id = getattr(model_version_obj, "run_id", None)
    
    # Parse run_id from source if not available
    if not run_id and model_version_obj.source and "runs:/" in model_version_obj.source:
        parts = model_version_obj.source.split("runs:/", 1)[-1].split("/", 1)
        run_id = parts[0]
    
    # Load the pyfunc model by version number
    print(f"Loading model version {model_version} (run: {run_id})...")
    pyfunc_model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{model_version}"
    )
    
    # Get metadata from run artifacts
    metadata = {"model_version": model_version, "run_id": run_id}
    
    # Add model version tags to metadata
    if model_version_obj.tags:
        metadata["tags"] = dict(model_version_obj.tags)
    
    if run_id:
        try:
            # Try to download metadata artifact
            for artifact_name in ("model_metadata", "metadata.json", "metadata"):
                try:
                    local_path = client.download_artifacts(run_id, artifact_name)
                    candidate = local_path
                    if os.path.isdir(local_path):
                        candidate = os.path.join(local_path, os.listdir(local_path)[0]) if os.listdir(local_path) else None
                    if candidate and os.path.exists(candidate):
                        with open(candidate, "r") as fh:
                            training_metadata = json.load(fh)
                            metadata.update(training_metadata)
                            break
                except Exception:
                    continue
        except Exception:
            pass
        
        # Fallback to run params/tags
        if "git_commit_hash" not in metadata:
            try:
                run = client.get_run(run_id)
                metadata["params"] = dict(run.data.params or {})
                metadata["run_tags"] = dict(run.data.tags or {})
            except Exception:
                pass
    
    return ModelWithMetadata(pyfunc_model, metadata, model_version=model_version, run_id=run_id)
