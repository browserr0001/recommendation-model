import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time
import pandas as pd
import pickle
import json
import requests
import tqdm

import pyarrow.parquet as pq

def read_data(path_prefix='data/'):
    # check this folder contains necessary parquet files
    if not (os.path.exists(f'{path_prefix}meta/movies.parquet')) or \
            not (os.path.exists(f'{path_prefix}meta/users_new.parquet')) or \
            not (os.path.exists(f'{path_prefix}ratings/ratings.parquet')) or \
            not (os.path.exists(f'{path_prefix}watches/watches.parquet')):
        raise FileNotFoundError("One or more required parquet files are missing in the specified path.")

    movies = pq.read_table(f'{path_prefix}meta/movies.parquet').to_pandas()
    users = pq.read_table(f'{path_prefix}meta/users_new.parquet').to_pandas()
    ratings = pq.read_table(f'{path_prefix}ratings/ratings.parquet').to_pandas()
    watches = pq.read_table(f'{path_prefix}watches/watches.parquet').to_pandas()

    selected_movie_cols = ['id', 'title', 'adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
    movies = movies[selected_movie_cols]
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['release_year'] = movies['release_date'].dt.year
    movies.rename(columns={'id': 'movie_id'}, inplace=True)
    

    def parse_name_field(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        parsed = value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        if isinstance(parsed, list):
            names = []
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    names.append(item["name"])
                elif isinstance(item, str):
                    names.append(item)
            return names
        if isinstance(parsed, dict) and "name" in parsed:
            return [parsed["name"]]
        return []
    movies['genres'] = movies['genres'].apply(parse_name_field)
    movies['production_companies'] = movies['production_companies'].apply(parse_name_field)
    movies['production_countries'] = movies['production_countries'].apply(parse_name_field)

    numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                           'vote_average', 'vote_count']
    for feature in numeric_features:
        movies[feature] = pd.to_numeric(movies[feature], errors='coerce')

    all_features = ['adult', 'budget', 'genres', 'original_language', 'overview',
                    'popularity', 'production_companies', 'production_countries', 
                    'release_date', 'release_year'] + numeric_features 
    assert all(feature in movies.columns for feature in all_features)
    # convert adult to boolean
    movies['adult'] = movies['adult'].astype(bool)
    return movies, users, ratings, watches


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
        self.age_bucket:int = None
        
        # Metadata for provenance tracking
        self.metadata = {
            'model_name': None,
            'model_tag': None,
            'git_commit_hash': None,
            'git_branch': None,
            'pipeline_version': None,
            'data_version': None,
            'training_params': {},
            'trained_at': None, 
            'data_start_timestamp': None,
            'data_end_timestamp': None
        }

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


    def create_user_profiles(self, users: pd.DataFrame, movie_features: np.ndarray, top_n=20, num_warm_users=10000):
        """Build user profiles with optimized cold-start handling"""
        print("Building user profiles...")
                
        # For existing users, create profiles based on their ratings
        user_profiles = {}
        
        # First handle users we have ratings for (fast path)
        rating_users = set(self.ratings_combined['user_id'].unique())
        print(f"Processing {min(num_warm_users, len(rating_users))} users with ratings...")
        
        groups = self.ratings_combined.groupby('user_id')
        user_prefs_map = {}
        # random sample of num_warm_users users from rating_users
        sample_users = np.random.choice(list(rating_users), size=min(num_warm_users, len(rating_users)), replace=False)
        for user_id in tqdm.tqdm(sample_users):
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
        user_prefs_map_df['age_group'] = (user_prefs_map_df['age'] // self.age_bucket) * self.age_bucket
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

    def fit(self, movies_df: pd.DataFrame, users_df: pd.DataFrame, ratings_df: pd.DataFrame, watches_df: pd.DataFrame, mid_rating_watch_over: float = 0.5, age_bucket: int = 10, metadata: dict = None, num_warm_users: int = 10000):
        """
        Train the content-based recommender
        
        Parameters:
        movies_df: DataFrame with movie metadata
        users_df: DataFrame with user metadata
        ratings_df: DataFrame with user ratings
        watches_df: Optional DataFrame with watch data
        metadata: Dictionary containing provenance information (git hash, data version, etc.)
        """
        # Validate input dataframes
        if movies_df.empty:
            raise ValueError("movies_df cannot be empty - movie metadata is required to train the model.")
        
        if users_df.empty:
            raise ValueError("users_df cannot be empty - user metadata is required for cold-start handling.")
        
        if ratings_df.empty and watches_df.empty:
            raise ValueError("At least one of ratings_df or watches_df must be non-empty to create user profiles.")
        
        t_start = time.time()
        self.movies_df = movies_df.copy()
        self.all_users = users_df.copy()
        self.age_bucket = age_bucket

        # Store training parameters and metadata for provenance
        if metadata:
            self.metadata.update(metadata)
        self.metadata['training_params'] = {
            'mid_rating_watch_over': mid_rating_watch_over,
            'age_bucket': age_bucket
        }
        self.metadata['trained_at'] = pd.Timestamp.now().isoformat()
        self.metadata['data_start_timestamp'] = min(
            watches_df['timestamp_start'].min() if not watches_df.empty else pd.Timestamp.max,
            ratings_df['timestamp'].min() if not ratings_df.empty else pd.Timestamp.max
        )        
        self.metadata['data_end_timestamp'] = max(watches_df['timestamp_end'].max() if not watches_df.empty else pd.Timestamp.min, ratings_df['timestamp'].max() if not ratings_df.empty else pd.Timestamp.min)

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

        self.create_user_profiles(users_df, movie_features, num_warm_users=num_warm_users)

        t_end = time.time()
        self.training_time = t_end - t_start
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def get_recommendations(self, user_id:int, top_n=10):
        """
        Get movie recommendations for a user
        
        Parameters:
        user_id: User ID to get recommendations for
        top_n: Number of recommendations to return
        
        Returns:
        recommendations: list of movie IDs
        inference_time: time taken for inference
        prediction_metadata: dictionary with model version and provenance info
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
                if user_age//self.age_bucket*self.age_bucket in self.all_age_groups and user_gender in self.all_genders:
                    ranked_movie_indices = self.group_profiles[user_gender][user_age//self.age_bucket*self.age_bucket]
                else:
                    ranked_movie_indices = self.group_profiles['Unknown'][-1]
            except:
                ranked_movie_indices = self.group_profiles['Unknown'][-1]
        
        movie_ids = self.movies_df.iloc[ranked_movie_indices]['movie_id'].values
        inference_time = time.time() - t_start
        
        # Return prediction metadata for logging
        prediction_metadata = {
            'model_name': self.metadata.get('model_name'),
            'model_tag': self.metadata.get('model_tag'),
            'git_commit_hash': self.metadata.get('git_commit_hash'),
            'pipeline_version': self.metadata.get('pipeline_version'),
            'data_version': self.metadata.get('data_version'),
            'data_start_timestamp': str(self.metadata.get('data_start_timestamp')),
            'data_end_timestamp': str(self.metadata.get('data_end_timestamp')),
            'training_params': self.metadata.get('training_params'),
            'trained_at': self.metadata.get('trained_at'),
            'inference_time': inference_time
        }
        
        return movie_ids[:top_n].tolist(), inference_time, prediction_metadata
    
    def get_metadata(self):
        """Return model metadata for provenance tracking"""
        return self.metadata.copy()

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

        return size

def train_model_full_data(movies, users, ratings, watches, path='app/model/results/content_based_model_full.pkl'):
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies, users, ratings, watches)
    pickle.dump(content_recommender, open(path, 'wb'))
    print(f"Model trained on full data and saved as '{path}'")
     # Print model metrics
    model_size_bytes = content_recommender.get_model_size()
    print(f"\nModel Metrics:")
    print(f"Full Training Time: {content_recommender.training_time:.2f} seconds")
    print(f"Full Model Size: {model_size_bytes / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # NOTE: To train the full model, read_data from data/ or whereever your full data is stored
    movies, users, ratings, watches = read_data('data_sample/')
    train_model_full_data(movies, users, ratings, watches, path='content_based_model_tiny.pkl')
    
