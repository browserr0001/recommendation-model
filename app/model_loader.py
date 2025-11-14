import os, pickle, threading
from app.model.src.content_based import ContentBasedRecommender

class ForceImportUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ContentBasedRecommender':
            # This always returns the right class, no matter the module path in pickle
            return ContentBasedRecommender
        return super().find_class(module, name)

_model_lock = threading.RLock()
_current_model = None
_current_path = None

def load_model_from_path(path: str):
    with open(path, "rb") as f:
        return ForceImportUnpickler(f).load()

def set_current_model(model, path):
    global _current_model, _current_path
    with _model_lock:
        _current_model = model
        _current_path = path

def get_current_model_and_path():
    with _model_lock:
        return _current_model, _current_path

def get_model(tag="Updated_Model") -> ContentBasedRecommender:
    print(f"loading model with tag: {tag}...")
    model_path = os.path.join(os.path.dirname(__file__), 'pkl_models', f'content_based_model_{tag}.pkl')
    model = load_model_from_path(model_path)
    set_current_model(model, model_path)
    return model
