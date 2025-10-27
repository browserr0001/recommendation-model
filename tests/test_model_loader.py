import os
import pytest
from unittest.mock import patch, mock_open
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
