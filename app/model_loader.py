import os
import pickle
from app.model.src.content_based import ContentBasedRecommender

class ForceImportUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ContentBasedRecommender':
            # This always returns the right class, no matter the module path in pickle
            return ContentBasedRecommender
        return super().find_class(module, name)

def get_model():
    print("loading model...")
    model_path = os.path.join(os.path.dirname(__file__), 'pkl_models', 'content_based_model_full.pkl')
    with open(model_path, 'rb') as f:
        model = ForceImportUnpickler(f).load()
    return model
