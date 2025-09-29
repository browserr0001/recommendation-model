## Content-based Recommendation Model
- The features included for training: movie meta data: ['adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
  - Genres has been one-hot encoded and other long categorical features (overview, production_companies, production_countries) has been treated simply as length. (we can change this if needed)
- The model handles cold start users by using the user metadata to find the most similar users and create the user profile based on the most similar user. Otherwise, if user metadata is not available, give generic recommendations.
- The model is tested on the old dataset. I'll think more about the accuracy metric to test it out.... (if anyone has good ideas that'll be really helpful)
- The code read from data  folder (which requires to download the data from the google drive). plz lmk if code does not run for you.
- This model runs with python 3.13.2, requirements are in requirements.txt


Current model outputs: 
```
Building movie profiles...
Building user profiles...
Processing 51487 users with ratings...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 51487/51487 [01:15<00:00, 684.83it/s]
Creating profiles for 48513 cold start users using batch processing...
Computing similarity matrix for cold-start users...
Creating cold-start profiles from similar users...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [03:14<00:00, 38.88s/it]
Created profiles for 100000 users total
Training completed in 297.32 seconds
Model trained on full data and saved as 'model/results/content_based_model_full.pkl'

Model Metrics:
Full Training Time: 297.32 seconds
Full Model Size: 106.43 MB
Created user profile from preprocessed user metadata
After User profile creation Time Elapsed: 0.1229 seconds
After Calculating Similarity Time Elapsed: 0.1284 seconds

Recommendations for new user 100000(cold start):
['whatever+1998', 'american+history+x+1998', '12+angry+men+1957', 'good+will+hunting+1997', 'the+color+purple+1985', 'philadelphia+1993', 'scent+of+a+woman+1992', 'an+adventure+in+space+and+time+2013', 'raging+bull+1980', 'the+twilight+of+the+golds+1996', 'sue+1997', 'the+joker+is+wild+1957', 'in+the+name+of+the+father+1993', 'there+will+be+blood+2007', 'awakenings+1990', 'the+diary+of+anne+frank+2009', 'apollo+13+1995', 'the+good+lie+2014', 'short+term+12+2013', 'lawn+dogs+1997']
Inference Time (cold start): 0.1316 seconds
```