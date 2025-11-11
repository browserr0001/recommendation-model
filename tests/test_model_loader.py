import os
import pytest
from unittest.mock import patch, mock_open
from app import model_loader


# def test_force_import_unpickler_resolves_class():
#     """Ensure ForceImportUnpickler correctly resolves the ContentBasedRecommender class."""
#     fake_data = b"cosmetic"
#     unpickler = model_loader.ForceImportUnpickler(open(os.devnull, "rb"))
#     cls = unpickler.find_class("model.src.content_based", "ContentBasedRecommender")
#     from app.model.src.content_based import ContentBasedRecommender
#     assert cls is ContentBasedRecommender


# @patch("builtins.open", new_callable=mock_open, read_data=b"fake_pkl_data")
# @patch.object(model_loader, "ForceImportUnpickler")
# def test_get_model_loads_with_custom_unpickler(mock_unpickler, mock_open_func):
#     """Simulate loading model via pickle to ensure no file access errors."""
#     fake_model = object()
#     mock_instance = mock_unpickler.return_value
#     mock_instance.load.return_value = fake_model

#     result = model_loader.get_model()

#     mock_open_func.assert_called_once()
#     mock_unpickler.assert_called_once()
#     mock_instance.load.assert_called_once()
#     assert result == fake_model


# --- Additional coverage for MLflow registry and ModelWithMetadata ---
import types
import pandas as pd


class DummyPyFuncModel:
    def __init__(self, predict_result):
        self._predict_result = predict_result

    def predict(self, model_input, top_n=20):
        return [self._predict_result]


class DummyModelVersion:
    def __init__(self, version, run_id, tags=None, source=None):
        self.version = version
        self.run_id = run_id
        self.tags = tags or {}
        self.source = source or f"runs:/{run_id}/model"


class DummyMlflowClient:
    def __init__(self, versions):
        self._versions = versions

    def get_latest_versions(self, name, stages):
        return self._versions

    def search_model_versions(self, filter_str):
        return self._versions

    def get_run(self, run_id):
        class Run:
            data = types.SimpleNamespace(params={"foo": "bar"}, tags={"baz": "qux"}, metrics={})

        return Run()

    def download_artifacts(self, run_id, artifact_name):
        return None

    def get_model_version(self, name, version):
        return self._versions[0]


@patch("app.model_loader.MlflowClient", autospec=True)
def test_get_model_by_tag_found(mock_mlflow_client):
    """Test get_model loads correct version by tag."""
    dummy_version = DummyModelVersion(version="42", run_id="abc123", tags={"model_tag": "Updated Model"})
    mock_mlflow_client.return_value = DummyMlflowClient([dummy_version])
    # Patch pyfunc model
    with patch("mlflow.pyfunc.load_model", return_value=DummyPyFuncModel((['rec1'], 0.1, {'meta': 'data'}))):
        model = model_loader.get_model(tag="Updated Model")
        assert hasattr(model, "predict")
        assert hasattr(model, "get_metadata")
        recs, inf_time, meta = model.predict(123, top_n=1)
        assert recs == ['rec1']
        assert meta == {'meta': 'data'}
        md = model.get_metadata()
        assert md["model_version"] == "42"
        assert md["run_id"] == "abc123"
        assert "tags" in md


@patch("app.model_loader.MlflowClient", autospec=True)
def test_get_model_by_tag_not_found_fallback(mock_mlflow_client):
    """Test get_model falls back to latest Production if tag not found."""
    dummy_version = DummyModelVersion(version="99", run_id="run99", tags={})
    mock_mlflow_client.return_value = DummyMlflowClient([dummy_version])
    with patch("mlflow.pyfunc.load_model", return_value=DummyPyFuncModel((['rec2'], 0.2, {'meta': 'fallback'}))):
        model = model_loader.get_model(tag="NonexistentTag")
        recs, inf_time, meta = model.predict(456, top_n=2)
        assert recs == ['rec2']
        assert meta == {'meta': 'fallback'}
        md = model.get_metadata()
        assert md["model_version"] == "99"
        assert md["run_id"] == "run99"


@patch("app.model_loader.MlflowClient", autospec=True)
def test_model_with_metadata_predict_and_metadata(mock_mlflow_client):
    """Test ModelWithMetadata.predict and get_metadata methods."""
    dummy_version = DummyModelVersion(version="1", run_id="run1", tags={})
    mock_mlflow_client.return_value = DummyMlflowClient([dummy_version])
    with patch("mlflow.pyfunc.load_model", return_value=DummyPyFuncModel((['recX'], 0.3, {'meta': 'info'}))):
        model = model_loader.get_model()
        recs, inf_time, meta = model.predict(789, top_n=3)
        assert recs == ['recX']
        assert meta == {'meta': 'info'}
        md = model.get_metadata()
        assert md["model_version"] == "1"
        assert md["run_id"] == "run1"
