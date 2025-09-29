## Content-based Recommendation Model
- The features included for training: movie meta data: ['adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
  - Genres has been one-hot encoded and other long categorical features (overview, production_companies, production_countries) has been treated simply as length. (we can change this if needed)
- The model handles cold start users by using the user metadata to find 5 similar users and create the user profile based on those 5 user. Otherwise, if user metadata is not available, give generic recommendations.
- The model is tested on the old dataset. I'll think more about the accuracy metric to test it out.... (if anyone has good ideas that'll be really helpful)
- The code read from data  folder (which requires to download the data from the google drive). plz lmk if code does not run for you.
- This model runs with python 3.13.2, requirements are in requirements.txt


Current model outputs: 
```
Building movie profiles...
Building user profiles...
Processing 51487 users with ratings...
100%|██████████████████████████████████████████████████████████████████████████████| 51487/51487 [01:25<00:00, 601.76it/s]
Creating profiles for 48513 cold start users using batch processing...
Computing similarity matrix for cold-start users...
Creating cold-start profiles from similar users...
100%|███████████████████████████████████████████████████████████████████████████████████████| 5/5 [03:14<00:00, 38.81s/it]
Created profiles for 100000 users total
Training completed in 304.63 seconds
Model trained on full data and saved as 'model/results/content_based_model_full.pkl'

Model Metrics:
Full Training Time: 304.63 seconds
Full Model Size: 106.43 MB
After User profile creation Time Elapsed: 0.0000 seconds
After Calculating Similarity Time Elapsed: 0.0142 seconds


Recommendations for new user 46052(cold start):
['head+games+2012', 'manufacturing+dissent+2007', 'riot+on+2004', 'the+whale+2011', 'mutantes+2009', 'under+our+skin+2008', 'welcome+to+macintosh+2008', 'confessions+of+a+burning+man+2003', 'the+living+sea+1995', 'mount+st.+elias+2009', 'the+rock-afire+explosion+2009', 'brutal+beauty+tales+of+the+rose+city+rollers+2010', 'dancing+outlaw+ii+jesco+goes+to+hollywood+1999', 'behind+the+burly+q+2010', 'mission+to+mir+1997', 'filming+othello+1978', 'addicted+to+plastic+2008', 'armbryterskan+frn+ensamheten+2004', 'the+jeffrey+dahmer+files+2013', 'fame+high+2012']
Inference Time (cold start): 0.0204 seconds
```