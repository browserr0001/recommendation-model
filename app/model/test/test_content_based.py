# !!! This script just to test if the tiny model works. 
# Please change to use unittest when developing testcases. !!!

import unittest
import pickle
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.content_based import ContentBasedRecommender, read_data


model = pickle.load(open('app/model/test/content_based_model_tiny.pkl', 'rb'))
print(model.get_recommendations(100))