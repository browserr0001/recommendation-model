import os
import pickle
from app.model.src.content_based import ContentBasedRecommender

class ForceImportUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ContentBasedRecommender':
            # Always return the current class, regardless of pickle module path
            return ContentBasedRecommender
        return super().find_class(module, name)

def get_model():
    print("loading model...")

    env_model_path = os.environ.get("MODEL_PATH")

    if env_model_path:
        model_path = env_model_path
    else:
        # Fallback to the original path you already use
        model_path = os.path.join(
            os.path.dirname(__file__),
            'pkl_models',
            'content_based_model_full.pkl'
        )

    print(f"Using model at: {model_path}")
    with open(model_path, 'rb') as f:
        model = ForceImportUnpickler(f).load()
    return model
