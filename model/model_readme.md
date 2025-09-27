## Content-based Recommendation Model
- The features included for training: movie meta data: ['adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
  - Genres has been one-hot encoded and other long categorical features (overview, production_companies, production_countries) has been treated simply as length. (we can change this if needed)
- The model handles cold start users by using the user metadata to find 5 similar users and create the user profile based on those 5 user. Otherwise, if user metadata is not available, give generic recommendations.
- The model is tested on the old dataset. I'll think more about the accuracy metric to test it out.... (if anyone has good ideas that'll be really helpful)
- The code read from data  folder (which requires to download the data from the google drive). plz lmk if code does not run for you.
- This model runs with python 3.13.2, requirements are in requirements.txt


Current model outputs: 
```
Train-test Split timestamp: 2025-09-23 02:19:06
Train ratings: (58968, 4), Test ratings: (8028, 5)
Train users: (30072, 4), Train movies: (14606, 16)
Test users: 7902
Processed 1000 users...ons for user 18847 999/7902
Processed 2000 users...ons for user 37695 1999/7902
Processed 3000 users...ons for user 56759 2999/7902
Processed 4000 users...ons for user 75607 3999/7902
Processed 5000 users...ons for user 94684 4999/7902
Processed 6000 users...ons for user 113513 5999/7902
Processed 7000 users...ons for user 132795 6999/7902
Overall Metrics:
Coverage: 71.7856%
Diversity: 100.0000%
New users in test set: 5481/7902

Model Metrics:
Training Time: 41.83 seconds
Model Size: 33.98 MB


Building movie profiles...
Building user profiles...
Training completed in 75.59 seconds
Model trained on full data and saved as 'model/results/content_based_model_full.pkl'

Model Metrics:
Full Training Time: 75.59 seconds
Full Model Size: 64.61 MB
```