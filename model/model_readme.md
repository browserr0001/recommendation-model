

## Content-based Recommendation Model
- The features included for training: movie meta data: ['adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
  - Genres has been one-hot encoded and other long categorical features (overview, production_companies, production_countries) has been treated simply as length. (we can change this if needed)
- The model handles cold start users by using the user metadata to find 5 similar users and create the user profile based on those 5 user. Otherwise, if user metadata is not available, give generic recommendations.
- The model is tested on the old dataset. I'll think more about the accuracy metric to test it out.... (if anyone has good ideas that'll be really helpful)
- The code read from data  folder (which requires to download the data from the google drive). plz lmk if code does not run for you.
- This model runs with python 3.13.2, requirements are in requirements.txt