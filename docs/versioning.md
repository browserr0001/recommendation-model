# Versioning Strategy

This project uses robust versioning for both data and models to ensure reproducibility, traceability, and collaboration.

## Data Versioning

- **Location:** All raw and processed data is stored under `data-pull/data/`.
- **Tool:** [DVC](https://dvc.org/) is used to track, version, and share data files.
- **How it works:**
	- Each data file (e.g., `ratings.parquet`, `watches.parquet`, `movies.parquet`, `users_new.parquet`) is tracked by DVC.
	- DVC creates `.dvc` files for each data artifact, storing metadata and hashes.
	- Data changes are committed to Git alongside their `.dvc` files, enabling reproducible pipelines.
	- Data can be pulled/pushed to remote storage for team collaboration.

## Model Versioning

- **Location:** Trained models are saved in `app/pkl_models/` as versioned `.pkl` files (e.g., `content_based_model_Initial_Model.pkl`, `content_based_model_Updated_Model.pkl`).
- **Tool:** DVC tracks model files as pipeline outputs, and Git records pipeline state in `dvc.lock`.
- **How it works:**
	- Each model is saved with a unique tag in the filename, reflecting its training context (e.g., `Initial_Model`, `Updated_Model`).
	- DVC pipeline stages in `dvc.yaml` define how models are trained and where outputs are stored.
	- Model files are versioned by DVC and changes are committed to Git via `dvc.lock` and `params.yaml`.
	- Model provenance (training parameters, git commit, data version) is embedded in the model metadata.

## Pipeline Versioning

- **Location:** Pipeline configuration is managed in `dvc.yaml` and `params.yaml`.
- **How it works:**
	- Each pipeline stage (e.g., `train_initial_model`, `train_updated_model`) specifies dependencies, parameters, and outputs.
	- Changes to pipeline logic or parameters are tracked in Git.
	- The `dvc.lock` file records the exact versions of data and code used for each run.

## Best Practices

- Always commit `.dvc` files, `dvc.lock`, and `params.yaml` to Git after running DVC commands.
- Use `dvc push` and `dvc pull` to sync data and models with remote storage.
- Tag model files clearly to reflect their training context and provenance.
- Use the embedded metadata in model files for audit and reproducibility.

## Example Workflow

1. Update or add new data files in `data-pull/data/`.
2. Run `dvc add` or update pipeline stages to track new data.
3. Commit changes to Git.
4. Retrain models using DVC pipeline (`dvc repro train_updated_model`).
5. DVC tracks the new model file in `app/pkl_models/` and updates `dvc.lock`.
6. Push data and models to remote storage with `dvc push`.
7. Commit pipeline and parameter changes to Git.

---
This versioning approach ensures that every model and dataset can be traced back to its source, parameters, and code version, supporting reproducible machine learning and robust collaboration.
