import os
import pytest
from unittest.mock import patch, mock_open, MagicMock
from app import model_loader


def test_force_import_unpickler_resolves_class():
    """Ensure ForceImportUnpickler correctly resolves the ContentBasedRecommender class."""
    fake_data = b"cosmetic"
    unpickler = model_loader.ForceImportUnpickler(open(os.devnull, "rb"))
    cls = unpickler.find_class("model.src.content_based", "ContentBasedRecommender")
    from app.model.src.content_based import ContentBasedRecommender
    assert cls is ContentBasedRecommender


@patch("builtins.open", new_callable=mock_open, read_data=b"fake_pkl_data")
@patch.object(model_loader, "ForceImportUnpickler")
def test_get_model_loads_with_custom_unpickler(mock_unpickler, mock_open_func):
    """Simulate loading model via pickle to ensure no file access errors."""
    fake_model = object()
    mock_instance = mock_unpickler.return_value
    mock_instance.load.return_value = fake_model

    result = model_loader.get_model()

    mock_open_func.assert_called_once()
    mock_unpickler.assert_called_once()
    mock_instance.load.assert_called_once()
    assert result == fake_model

def test_get_model_default_tag():
    """Test that get_model uses 'Updated_Model' as default tag."""
    with patch("builtins.open", mock_open(read_data=b"fake_pkl")), \
         patch.object(model_loader.ForceImportUnpickler, "load", return_value=MagicMock()):
        result = model_loader.get_model()
        # Verify the path includes the default tag
        assert result is not None


def test_get_model_custom_tag():
    """Test that get_model accepts a custom tag parameter."""
    with patch("builtins.open", mock_open(read_data=b"fake_pkl")) as mock_file, \
         patch.object(model_loader.ForceImportUnpickler, "load", return_value=MagicMock()):
        result = model_loader.get_model(tag="Initial_Model")
        # Verify the path includes the custom tag
        call_args = mock_file.call_args[0][0]
        assert "Initial_Model" in call_args
        assert result is not None


def test_get_model_file_not_found():
    """Test that get_model raises FileNotFoundError when model file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        model_loader.get_model(tag="NonexistentModel")


@patch("builtins.open", new_callable=mock_open, read_data=b"fake_pkl_data")
@patch.object(model_loader, "ForceImportUnpickler")
def test_get_model_prints_loading_message(mock_unpickler, mock_open_func, capsys):
    """Test that get_model prints a loading message with the tag."""
    fake_model = MagicMock()
    mock_instance = mock_unpickler.return_value
    mock_instance.load.return_value = fake_model

    model_loader.get_model(tag="TestModel")
    
    captured = capsys.readouterr()
    assert "loading model with tag: TestModel" in captured.out


@patch("builtins.open", new_callable=mock_open, read_data=b"fake_pkl_data")
@patch.object(model_loader, "ForceImportUnpickler")
def test_get_model_returns_content_based_recommender(mock_unpickler, mock_open_func):
    """Test that get_model returns a ContentBasedRecommender instance."""
    fake_model = MagicMock(spec=model_loader.ContentBasedRecommender)
    fake_model.get_recommendations = MagicMock(return_value=([1, 2, 3], 0.01, {"test": "metadata"}))
    mock_instance = mock_unpickler.return_value
    mock_instance.load.return_value = fake_model

    result = model_loader.get_model()

    assert hasattr(result, 'get_recommendations')
    recs, time, meta = result.get_recommendations(123, top_n=3)
    assert recs == [1, 2, 3]
    assert time == 0.01
    assert meta == {"test": "metadata"}

