import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from read_data import read_data
import pickle
import json
import requests
import tqdm

class ContentBasedRecommender:
    def __init__(self):
        # This movie_profiles is a 2D numpy array where each row corresponds to a movie and each column to a feature
        self.movie_profiles:np.ndarray = None
        # User profiles is a mapping of user_id to list of movie indices ranked by similarity
        self.user_profiles:dict = None
        # To handle cold start users, we will create group profiles based on demographics
        self.group_profiles:dict = None
        self.all_genders:set = None
        self.all_age_groups:set = None
        # Use this movies_df to find the movie_id corresponding to an index in movie_profiles and user_profiles
        self.movies_df:pd.DataFrame = None
        self.all_users:pd.DataFrame = None
        self.ratings_combined:pd.DataFrame = None
        self.cold_start_user_pipeline:Pipeline = None

    def create_movie_profiles(self, movies: pd.DataFrame) -> np.ndarray:
        # Build movie profiles
        print("Building movie profiles...")
        n_movies = len(movies)
        
        # Numeric features to include
        numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                           'vote_average', 'vote_count']
        
        # Initialize feature arrays
        numeric_feature_matrix = np.zeros((n_movies, len(numeric_features)))
        
        # Extract and normalize numeric features
        for i, feature in enumerate(numeric_features):
            feature_values = movies[feature].values
            feature_values = np.nan_to_num(feature_values)
            # Normalize if not all zeros to avoid division by zero
            if np.sum(np.abs(feature_values)) > 0:
                feature_values = feature_values / np.max(np.abs(feature_values))
                
            numeric_feature_matrix[:, i] = feature_values
        
        # Process categorical features
        
        # Adult feature (boolean)
        adult_feature = np.zeros((n_movies, 1))
        adult_feature[:, 0] = movies['adult'].fillna(False).astype(int).values
        
        # Original language (one-hot encoding)
        languages = movies['original_language'].fillna('unknown').values
        unique_languages = np.unique(languages)
        language_features = np.zeros((n_movies, len(unique_languages)))
        lang_to_idx = {lang: i for i, lang in enumerate(unique_languages)}
        
        for i, lang in enumerate(languages):
            language_features[i, lang_to_idx[lang]] = 1
        
        # Release date - extract year and normalize
        release_years = movies['release_year'].fillna(0).values.reshape(-1, 1)
        
        # Normalize years
        if np.max(release_years) > np.min(release_years):
            release_years = (release_years - np.min(release_years)) / (np.max(release_years) - np.min(release_years))
        
        # Extract genres as a list of features
        genre_lists = movies['genres'].apply(lambda x: [] if x is None else x)
        
        # Create one-hot encoding for genres
        all_genres = set()
        for genres in genre_lists:
            if isinstance(genres, list):
                all_genres.update(genres)
                
        # Create genre features matrix
        n_genres = len(all_genres)
        genre_features = np.zeros((n_movies, n_genres))
        
        # Map genre names to column indices
        genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
        
        # Fill the genre features matrix
        for i, genres in enumerate(genre_lists):
            if isinstance(genres, list):
                for genre in genres:
                    if genre in genre_to_idx:
                        genre_features[i, genre_to_idx[genre]] = 1

        # Process text features - convert overview to a simple length feature
        overview_length = np.zeros((n_movies, 1))

        for i, overview in enumerate(movies['overview'].fillna('')):
            if isinstance(overview, str):
                overview_length[i, 0] = min(1.0, len(overview) / 1000)  # Normalize to [0,1]
        
        # Process production companies and countries - use count as a feature
        company_counts = np.zeros((n_movies, 1))
        country_counts = np.zeros((n_movies, 1))

        for i, companies in enumerate(movies['production_companies']):
            if isinstance(companies, list):
                company_counts[i, 0] = min(1.0, len(companies) / 10)  # Normalize to [0,1]

        for i, countries in enumerate(movies['production_countries']):
            if isinstance(countries, list):
                country_counts[i, 0] = min(1.0, len(countries) / 5)  # Normalize to [0,1]
        
        # Concatenate all feature matrices
        movie_features = np.hstack([
            genre_features,
            numeric_feature_matrix,
            adult_feature,
            language_features,
            release_years,
            overview_length,
            company_counts,
            country_counts
        ])
        return movie_features


    def create_user_profiles(self, users: pd.DataFrame, movie_features: np.ndarray, top_n=20):
        """Build user profiles with optimized cold-start handling"""
        print("Building user profiles...")
        
        # Create user metadata encoder for cold start
        user_features = ['age', 'gender']
        self.cold_start_user_pipeline = Pipeline([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
            ('scaler', StandardScaler())
        ])
        
        # Extract user features for training the cold start encoder
        user_meta_features = users[user_features]
        self.cold_start_user_pipeline.fit(user_meta_features)
        
        # For existing users, create profiles based on their ratings
        user_profiles = {}
        
        # First handle users we have ratings for (fast path)
        rating_users = set(self.ratings_combined['user_id'].unique())
        print(f"Processing {len(rating_users)} users with ratings...")
        
        groups = self.ratings_combined.groupby('user_id')
        user_prefs_map = {}
        for user_id in tqdm.tqdm(rating_users):
            group = groups.get_group(user_id)
            # Get the movies this user has rated
            rated_movie_indices = []
            for movie_id in group['movie_id']:
                movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index
                if len(movie_idx) > 0:
                    rated_movie_indices.append(movie_idx[0])
            
            if rated_movie_indices:
                # Get the genres of movies this user has rated
                user_prefs = np.zeros(movie_features.shape[1])
                ratings = np.array(group['rating'])
                
                # Weight the genres by the ratings
                for i, idx in enumerate(rated_movie_indices):
                    user_prefs += ratings[i] * self.movie_profiles[idx]
                    
                # Normalize
                if np.sum(user_prefs) > 0:
                    user_prefs = user_prefs / np.sum(user_prefs)
                    
                # precalculate the recommendations for this user
                similarity_scores = cosine_similarity([user_prefs], self.movie_profiles)[0]
                user_prefs_map[user_id] = user_prefs
                # save the indices of top 20 recommendations ranked by similarity
                ranked_movie_indices = np.argsort(similarity_scores)[::-1]
                user_profiles[user_id] = ranked_movie_indices[:top_n]
        self.user_profiles = user_profiles
        user_prefs_map_df = pd.DataFrame.from_dict(user_prefs_map, orient='index')
        # add age and gender to user_prefs_map_df by merging with users
        user_prefs_map_df = user_prefs_map_df.merge(users[['user_id', 'age', 'gender']], left_index=True, right_on='user_id', how='left')

        # Now handle cold start users more efficiently
        print("Creating cold-start group profiles...")
        # Group self.all_users by gender and age (in decades)
        # For all users in each group, assign the average profile of that group
        # This will help in cold start for users not in the preprocessed set
        group_profiles = {}
        # Get the age_group in user_prefs_map_df
        user_prefs_map_df['age_group'] = (user_prefs_map_df['age'] // 10) * 10
        user_prefs_map_df['age_group'] = user_prefs_map_df['age_group'].fillna(-1).astype(int)
        user_prefs_map_df['gender'] = user_prefs_map_df['gender'].fillna('Unknown')
        # group user_prefs_map_df by gender and age_group and get the mean of user_prefs columns
        prefs_groups = user_prefs_map_df.groupby(['gender', 'age_group'])
        for name, group in prefs_groups:
            if len(group) > 0:
                group_prefs = group.drop(columns=['user_id', 'age', 'gender', 'age_group']).mean().values
                if np.sum(group_prefs) > 0:
                    group_prefs = group_prefs / np.sum(group_prefs)
                group_recs = np.argsort(cosine_similarity([group_prefs], self.movie_profiles)[0])[::-1]
                group_profiles[name] = group_recs[:top_n]
        # reorganize group_profiles to {'gender': {'age_group': recs}}
        reorganized_group_profiles = {}
        for (gender, age_group), recs in group_profiles.items():      
            if gender not in reorganized_group_profiles:
                reorganized_group_profiles[gender] = {}
            reorganized_group_profiles[gender][age_group] = recs
        self.group_profiles = reorganized_group_profiles

        # Add a default profile for unknown demographics (movies ranked by overall popularity)
        self.group_profiles['Unknown'] = {}
        self.group_profiles['Unknown'][-1] = np.argsort(self.movies_df['popularity'].values)[::-1][:top_n]

        self.all_genders = set(self.group_profiles.keys())
        self.all_age_groups = set()
        for gender in self.group_profiles:
            self.all_age_groups.update(self.group_profiles[gender].keys())

        # print(f"Created profiles for {len(user_profiles)} users total")
        return 

    def fit(self, movies_df: pd.DataFrame, users_df: pd.DataFrame, ratings_df: pd.DataFrame, watches_df: pd.DataFrame, mid_rating_watch_over: float = 0.5):
        """
        Train the content-based recommender
        
        Parameters:
        movies_df: DataFrame with movie metadata
        users_df: DataFrame with user metadata
        ratings_df: DataFrame with user ratings
        watches_df: Optional DataFrame with watch data
        """
        t_start = time.time()
        self.movies_df = movies_df.copy()
        self.all_users = users_df.copy()

        # merge watches_df with movies_df and calculate the percent watched 
        # If the user watched over half of the movie, it is automatically counted as a mid rating
        percent_watched = pd.merge(watches_df[['movie_id', 'user_id', 'timestamp_end', 'minutes_watched']], movies_df[['movie_id', 'runtime']], on='movie_id')
        percent_watched.rename(columns={'timestamp_end': 'timestamp'}, inplace=True)
        percent_watched = percent_watched[percent_watched['minutes_watched']/percent_watched['runtime']>=mid_rating_watch_over]
        percent_watched['rating'] = 3
        
        # Select required columns from watches and ratings
        watch_subset = percent_watched[['timestamp', 'user_id', 'movie_id', 'rating']]
        overall_ratings = pd.concat([watch_subset, ratings_df])
        overall_ratings = overall_ratings.sort_values('timestamp')
        self.ratings_combined = overall_ratings.groupby(['user_id', 'movie_id']).last().reset_index()

        movie_features = self.create_movie_profiles(self.movies_df)
                        
        # Store the movie profiles
        self.movie_profiles = movie_features
        self.num_total_features = movie_features.shape[1]

        self.create_user_profiles(users_df, movie_features)

        t_end = time.time()
        self.training_time = t_end - t_start
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def get_recommendations(self, user_id, top_n=10, exclude_seen=True):
        """
        Get movie recommendations for a user
        
        Parameters:
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return
        exclude_seen: Whether to exclude movies the user has already rated
        
        Returns:
        recommendations: DataFrame with movie recommendations
        """
        t_start = time.time()
        if user_id in self.user_profiles.keys():
            # Existing user, get top_n recommendations from precomputed list
            ranked_movie_indices = self.user_profiles[user_id]
        else:
            # Cold start users
            try: 
                user_age = self.all_users[self.all_users['user_id'] == user_id]['age'][0]
                user_gender = self.all_users[self.all_users['user_id'] == user_id]['gender'][0]
                if user_age//10*10 in self.all_age_groups and user_gender in self.all_genders:
                    ranked_movie_indices = self.group_profiles[user_gender][user_age//10*10]
                else:
                    ranked_movie_indices = self.group_profiles['Unknown'][-1]
            except:
                ranked_movie_indices = self.group_profiles['Unknown'][-1]
        
        movie_ids = self.movies_df.iloc[ranked_movie_indices]['movie_id'].values
        return movie_ids[:top_n].tolist(), time.time() - t_start

    def get_model_size(self):
        """
        Calculate the approximate memory footprint of the model
        """
        size = 0
        
        # Movie profiles size
        if self.movie_profiles is not None:
            size += self.movie_profiles.nbytes
            
        # User profiles size
        if self.user_profiles is not None:
            # Approximate since it's a dictionary
            size += sum(profile.nbytes for profile in self.user_profiles.values())
            size += len(self.user_profiles) * 8  # Dictionary overhead
            
        # Pipeline size (approximate)
        if self.cold_start_user_pipeline is not None:
            # Rough estimate for the pipeline
            size += 10000  # Base estimate
            
        return size


def train_test_split(ratings_df, movies_df, users_df, watches_df, test_size=0.2, mid_rating_watch_over=0.5):
    """
    Split the ratings data and watches data into training and test sets based on timestamp.
    For movies and users, include only those present in the training set.
    Parameters:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie metadata
        users_df: DataFrame with user metadata
        watches_df: DataFrame with user watch history
        test_size: Proportion of data to use for testing
    Returns:
        train_ratings: Training set of ratings
        test_ratings: Test set of ratings
        train_movies: Movies present in the training set
        train_users: Users present in the training set
        train_watches: Training set of watches
        test_watches: Test set of watches
    
    """
    # Get the split timestamp based on percentage of all timestamps
    ratings_df = ratings_df.sort_values('timestamp').reset_index(drop=True)
    n_total = len(ratings_df)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    train_ratings = ratings_df.iloc[:n_train].reset_index(drop=True)
    test_ratings = ratings_df.iloc[n_train:].reset_index(drop=True)

    split_timestamp = ratings_df.iloc[n_train]['timestamp']
    print(f"Train-test Split timestamp: {split_timestamp}")

    # Get users and movies in the training set
    train_user_ids = train_ratings['user_id'].unique()
    train_movie_ids = train_ratings['movie_id'].unique()

    train_users = users_df[users_df['user_id'].isin(train_user_ids)].reset_index(drop=True)
    train_movies = movies_df[movies_df['movie_id'].isin(train_movie_ids)].reset_index(drop=True)
    
    train_watches = watches_df[watches_df['timestamp_start'] <= split_timestamp].reset_index(drop=True)
    test_watches = watches_df[watches_df['timestamp_start'] > split_timestamp].reset_index(drop=True)

    test_watch_subset = test_watches[['timestamp_start', 'user_id', 'movie_id']]
    test_ratings = pd.concat([test_watch_subset, test_ratings]).groupby(['user_id', 'movie_id']).last().reset_index()
    test_users = test_ratings['user_id'].unique()
    print(f"Train ratings: {train_ratings.shape}, Test ratings: {test_ratings.shape}")
    print(f"Train users: {train_users.shape}, Train movies: {train_movies.shape}")

    return train_ratings,  train_movies, train_users, train_watches, test_ratings, test_users

def calculate_accuracy(user_id, recommendations, test_ratings):
    """
    Calculate the percentage of recommended movies shows in test_ratings for the user_id.
    """
    if user_id not in test_ratings['user_id'].values:
        return 0.0
    
    user_test_movies = test_ratings[test_ratings['user_id'] == user_id]['movie_id'].values

    
    if len(user_test_movies) == 0:
        return 0.0
    
    hits = np.isin(recommendations, user_test_movies).sum()
    accuracy = hits / len(recommendations)

    return accuracy


def calculate_diversity(rec_ids, movies_df):
    genres_seen = set()
    for mid in rec_ids:
        row = movies_df[movies_df["movie_id"] == mid]
        if not row.empty:
            genres = row.iloc[0]["genres"]
            genres_seen.update(set(genres))
    return len(genres_seen) / (len(rec_ids) + 1e-9)

def calculate_coverage(all_rec_lists, all_movie_ids):
    recommended_movies = set()
    for recs in all_rec_lists:
        recommended_movies.update(recs)
    return len(recommended_movies) / len(all_movie_ids)


def evaluate_precision(content_recommender:ContentBasedRecommender, users_subset, test_df, movies, k=20):

    metrics = {"precision": [], "recall": [], "ndcg": [], "accuracy": [], "diversity": []}
    all_rec_lists = []
    test_relevant = test_df["movie_id"].tolist()
    print(f"Total test relevant movies: {len(test_relevant)}")

    for user_id in users_subset:
        test_watched = test_df[test_df["user_id"] == user_id]["movie_id"].tolist()

        if not test_watched:
            continue

        rec_ids, _ = content_recommender.get_recommendations(user_id, top_n=k, exclude_seen=True)
        
        all_rec_lists.append(rec_ids)

        hits_relevant = len(set(rec_ids) & set(test_relevant))
        hits_watched = len(set(rec_ids) & set(test_watched))

        metrics["precision"].append(hits_relevant / k)
        metrics["recall"].append(hits_relevant / len(test_relevant) if test_relevant else 0)

        dcg = sum([1/np.log2(i+2) for i, mid in enumerate(rec_ids) if mid in test_relevant])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(test_relevant), k))])
        metrics["ndcg"].append(dcg/idcg if idcg > 0 else 0)

        metrics["accuracy"].append(hits_watched / k)
        metrics["diversity"].append(calculate_diversity(rec_ids, movies))

    results = {m: float(np.mean(v)) for m, v in metrics.items() if v}
    results["coverage"] = calculate_coverage(all_rec_lists, movies["movie_id"].unique())
    return results


def calculate_recommendation_metrics(user_id, recommendations, test_ratings, train_movies):
    """
    Calculate multiple recommendation quality metrics for a user
    
    Parameters:
    user_id: User ID to evaluate
    recommendations: List of recommended movie_ids
    test_ratings: DataFrame with test ratings/watches data
    
    Returns:
    dict: Dictionary of metrics including hit_rate, precision@k, recall, ndcg, and diversity
    """
    metrics = {}
    
    # If user not in test set, return zeros for all metrics
    if user_id not in test_ratings['user_id'].values:
        return {
            'hit_rate': 0.0,
            'precision': 0.0, 
            'recall': 0.0,
            'diversity': 0.0
        }
    
    # Get movies the user interacted with in the test set
    user_test_movies = test_ratings[test_ratings['user_id'] == user_id]['movie_id'].values
    
    if len(user_test_movies) == 0:
        return {
            'hit_rate': 0.0, 
            'precision': 0.0, 
            'recall': 0.0,
            'diversity': 0.0
        }
    
    # Calculate hit rate (original accuracy metric)
    hits = np.isin(recommendations, user_test_movies).sum()
    metrics['hit_rate'] = hits / len(recommendations)
    
    # Calculate precision
    metrics['precision'] = hits / len(recommendations)
    
    # Calculate recall
    metrics['recall'] = hits / len(user_test_movies)
    
    # Calculate recommendation diversity (using genres if available)
    if train_movies is not None:
        # Get the genres for recommended movies
        rec_genres = set()
        for movie_id in recommendations:
            movie_data = train_movies[train_movies['movie_id'] == movie_id]
            if not movie_data.empty and movie_data['genres'].iloc[0] is not None:
                genres = movie_data['genres'].iloc[0]
                if isinstance(genres, list):
                    rec_genres.update(genres)
        
        # Diversity is the number of unique genres divided by the average number of genres per movie
        avg_genres_per_movie = len(rec_genres) / len(recommendations) if len(recommendations) > 0 else 0
        metrics['diversity'] = len(rec_genres) / (avg_genres_per_movie * 10) if avg_genres_per_movie > 0 else 0
    else:
        metrics['diversity'] = 0.0
    

    return metrics

def calculate_overall_metrics(recommendations_list, train_movies):
    """
    Calculate the overall metrics for the recommendation system.

    Parameters:
    recommendations_list: List of lists of recommended movie_ids for multiple users
    all_movie_ids: Set or list of all movie_ids in the dataset

    Returns:
    dict: Dictionary containing overall metrics including coverage and diversity
    """
    # Calculate coverage
    total_num_movies = len(train_movies['movie_id'].unique())
    recommended_movies = set()
    for recs in recommendations_list:
        recommended_movies.update(recs)

    coverage = len(recommended_movies) / total_num_movies if total_num_movies > 0 else 0.0
    # Calculate diversity
    all_genres = set()
    recommended_genres = set()
    for recs in recommendations_list:
        for movie_id in recs:
            movie_data = train_movies[train_movies['movie_id'] == movie_id]
            if not movie_data.empty and movie_data['genres'].iloc[0] is not None:
                genres = movie_data['genres'].iloc[0]
                if isinstance(genres, list):
                    recommended_genres.update(genres)
    for genres in train_movies['genres'].dropna():
        if isinstance(genres, list):
            all_genres.update(genres)

    diversity = len(recommended_genres) / len(all_genres) if len(all_genres) > 0 else 0.0

    return {
        'coverage': coverage,
        'diversity': diversity
    }
 

def get_user_metadata(user_id, link="http://128.2.220.241:8080/user"):
    """
    Make get request to the link following structure: http://128.2.220.241:8080/user/23469
    """
    response = requests.get(f"{link}/{user_id}")
    if response.status_code == 200:
        return response.json()
    else:
        return None


def run_train_test(movies, users, ratings, watches, train=False, user_specific_test=False):
    train_ratings,  train_movies, train_users, train_watches, test_ratings, test_users = train_test_split(ratings, movies, users, watches, test_size=0.05)
    # pickle.dump(test_ratings, open('model/results/content_based_test_ratings.pkl', 'wb'))
    if user_specific_test:
        test_users = pd.read_csv('data/test_users_all.csv')['user_id'].unique()
    test_df = test_ratings[test_ratings['user_id'].isin(test_users)].reset_index(drop=True)
    print(f"Test users: {test_df['user_id'].nunique()}")
    
    if train: 
        content_recommender = ContentBasedRecommender()
        # Fit the model on the training data
        content_recommender.fit(train_movies, users, train_ratings, train_watches)
        pickle.dump(content_recommender, open('model/results/content_based_model.pkl', 'wb'))

    else:
        content_recommender = pickle.load(open('model/results/content_based_model.pkl', 'rb'))

    results = evaluate_precision(content_recommender, test_users, test_df, movies, k=20)
    print(f"\nEvaluation Results on Test Set:")
    print(results)
    
    # Print model metrics
    model_size_bytes = content_recommender.get_model_size()
    print(f"\nModel Metrics:")
    print(f"Training Time: {content_recommender.training_time:.2f} seconds")
    print(f"Model Size: {model_size_bytes / (1024*1024):.2f} MB")
    results['model_size_mb'] = model_size_bytes / (1024*1024)
    results['training_time_sec'] = content_recommender.training_time

    with open('model/results/content_based_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results, model_size_bytes, content_recommender.training_time


def train_model_full_data(movies, users, ratings, watches):
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies, users, ratings, watches)
    pickle.dump(content_recommender, open('model/results/content_based_model_full.pkl', 'wb'))
    print("Model trained on full data and saved as 'model/results/content_based_model_full.pkl'")

     # Print model metrics
    model_size_bytes = content_recommender.get_model_size()
    print(f"\nModel Metrics:")
    print(f"Full Training Time: {content_recommender.training_time:.2f} seconds")
    print(f"Full Model Size: {model_size_bytes / (1024*1024):.2f} MB")

if __name__ == "__main__":
    movies, users, ratings, watches = read_data('data/')
    # model = pickle.load(open('model/results/content_based_model_full.pkl', 'rb'))
    # run_train_test(movies, users, ratings, watches, train=True, user_specific_test=True)
    train_model_full_data(movies, users, ratings, watches)
