from model.src.content_based import read_data, train_test_split, run_train_test, get_recommendations, ContentBasedRecommender
import unittest

def test_run_train_test():
    movies, users, ratings, watches = read_data('data_sample/')
    results, model_size_bytes, training_time = run_train_test(movies, users, ratings, watches, train=True)
    assert results is not None
    assert model_size_bytes > 0
    assert training_time > 0
    print("test_run_train_test passed.")


def test_get_recommendations():
    # load a pre-trained model
    model = ContentBasedRecommender()
    model = model.load_model('model/results/content_based_model.pkl')

    user_id = 88011  # example user_id
    recommendations = get_recommendations(model, user_id, top_n=5)
    assert len(recommendations) == 5
    print("test_get_recommendations passed.")



if __name__ == "__main__":
    test_run_train_test()
    test_get_recommendations()