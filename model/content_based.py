import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from read_data import read_data

class ContentBasedRecommender:
    def __init__(self):
        self.movie_profiles = None
        self.user_profiles = None
        self.movies_df = None
        self.users_df = None
        self.ratings_combined = None
        self.user_encoder = None
        self.movie_encoder = None
        self.cold_start_user_pipeline = None
        
    def fit(self, movies_df, users_df, ratings_df, watches_df, mid_rating_watch_over=0.5):
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
        self.users_df = users_df.copy()
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


        # Build movie profiles
        print("Building movie profiles...")
        n_movies = len(self.movies_df)
        
        
        
        # Numeric features to include
        numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                           'vote_average', 'vote_count']
        
        # Initialize feature arrays
        numeric_feature_matrix = np.zeros((n_movies, len(numeric_features)))
        
        # Extract and normalize numeric features
        for i, feature in enumerate(numeric_features):
            feature_values = self.movies_df[feature].values
            feature_values = np.nan_to_num(feature_values)
            # Normalize if not all zeros to avoid division by zero
            if np.sum(np.abs(feature_values)) > 0:
                feature_values = feature_values / np.max(np.abs(feature_values))
                
            numeric_feature_matrix[:, i] = feature_values
        
        # Process categorical features
        
        # Adult feature (boolean)
        adult_feature = np.zeros((n_movies, 1))
        adult_feature[:, 0] = self.movies_df['adult'].fillna(False).astype(int).values
        
        # Original language (one-hot encoding)
        languages = self.movies_df['original_language'].fillna('unknown').values
        unique_languages = np.unique(languages)
        language_features = np.zeros((n_movies, len(unique_languages)))
        lang_to_idx = {lang: i for i, lang in enumerate(unique_languages)}
        
        for i, lang in enumerate(languages):
            language_features[i, lang_to_idx[lang]] = 1
        
        # Release date - extract year and normalize
        release_years = self.movies_df['release_year'].fillna(0).values.reshape(-1, 1)
        
        # Normalize years
        if np.max(release_years) > np.min(release_years):
            release_years = (release_years - np.min(release_years)) / (np.max(release_years) - np.min(release_years))
        
        # Extract genres as a list of features
        genre_lists = self.movies_df['genres'].apply(lambda x: [] if x is None else x)
        
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
        
        for i, overview in enumerate(self.movies_df['overview'].fillna('')):
            if isinstance(overview, str):
                overview_length[i, 0] = min(1.0, len(overview) / 1000)  # Normalize to [0,1]
        
        # Process production companies and countries - use count as a feature
        company_counts = np.zeros((n_movies, 1))
        country_counts = np.zeros((n_movies, 1))
        
        for i, companies in enumerate(self.movies_df['production_companies']):
            if isinstance(companies, list):
                company_counts[i, 0] = min(1.0, len(companies) / 10)  # Normalize to [0,1]
                
        for i, countries in enumerate(self.movies_df['production_countries']):
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
                        
        # Store the movie profiles
        self.movie_profiles = movie_features
        self.num_total_features = movie_features.shape[1]
        
        # Build user profiles
        print("Building user profiles...")
        
        # Create user metadata encoder for cold start
        user_features = ['age', 'occupation', 'gender']
        self.cold_start_user_pipeline = Pipeline([
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
            ('scaler', StandardScaler())
        ])
        
        # Extract user features for training the cold start encoder
        user_meta_features = self.users_df[user_features]
        self.cold_start_user_pipeline.fit(user_meta_features)
        
        # For existing users, create profiles based on their ratings
        user_profiles = {}
        
        # Use the combined ratings dataset that includes both explicit ratings and watch behavior
        for user_id, group in self.ratings_combined.groupby('user_id'):
            if user_id in self.users_df['user_id'].values:
                # Get the movies this user has rated
                rated_movie_indices = []
                movie_ids = []
                for movie_id in group['movie_id']:
                    movie_idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index
                    if len(movie_idx) > 0:
                        rated_movie_indices.append(movie_idx[0])
                        movie_ids.append(movie_id)
                
                if rated_movie_indices:
                    # Get the genres of movies this user has rated
                    user_genre_prefs = np.zeros(movie_features.shape[1])
                    ratings = np.array(group['rating'])
                    
                    # Weight the genres by the ratings
                    for i, idx in enumerate(rated_movie_indices):
                        user_genre_prefs += ratings[i] * self.movie_profiles[idx]
                        
                    # Normalize
                    if np.sum(user_genre_prefs) > 0:
                        user_genre_prefs = user_genre_prefs / np.sum(user_genre_prefs)
                        
                    user_profiles[user_id] = user_genre_prefs
        
        self.user_profiles = user_profiles
        
        t_end = time.time()
        self.training_time = t_end - t_start
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def create_user_profile_cold_start(self, user_data):
        """
        Create a user profile for a new user based on demographic information
        
        Parameters:
        user_data: dict with 'age', 'occupation', 'gender'
        
        Returns:
        user_profile: numpy array representing user preferences
        """
        # Convert user data to DataFrame
        if isinstance(user_data, dict):
            user_df = pd.DataFrame([user_data])
        else:
            user_df = pd.DataFrame([{
                'age': user_data.get('age', 25), 
                'occupation': user_data.get('occupation', 'other'),
                'gender': user_data.get('gender', 'M')
            }])
            
        # Transform user metadata into feature vector
        user_features = self.cold_start_user_pipeline.transform(user_df)
        
        # Find similar users in the training set
        user_similarities = cosine_similarity(
            user_features, 
            self.cold_start_user_pipeline.transform(self.users_df[['age', 'occupation', 'gender']])
        )[0]
        
        # Get top 5 similar users
        top_user_indices = np.argsort(user_similarities)[-5:]
        top_user_ids = self.users_df.iloc[top_user_indices]['user_id'].values
        
        # Create a profile based on similar users' profiles
        cold_start_profile = np.zeros(self.num_total_features)
        weights = user_similarities[top_user_indices]
        weights = weights / np.sum(weights)  # Normalize weights
        
        for i, user_id in enumerate(top_user_ids):
            if user_id in self.user_profiles:
                cold_start_profile += weights[i] * self.user_profiles[user_id]
        
        return cold_start_profile
    
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
        
        # Check if this is a known or new user
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            print(f"Generating recommendations for existing user {user_id}")
        else:
            # Cold start: get user metadata if available
            user_data = self.users_df[self.users_df['user_id'] == user_id]
            
            if len(user_data) > 0:
                user_meta = user_data.iloc[0].to_dict()
                user_profile = self.create_user_profile_cold_start(user_meta)
                print(f"Generating recommendations for cold start user {user_id} with metadata")
            else:
                # No metadata for this user, use default profile
                print(f"No metadata for user {user_id}. Using generic profile.")
                user_profile = np.ones(self.num_total_features) / self.num_total_features  # Equal preference for all features
        
        # Calculate similarity to each movie
        similarity_scores = cosine_similarity([user_profile], self.movie_profiles)[0]
        
        # Create a DataFrame with movie_ids and similarity scores
        recommendations = pd.DataFrame({
            'movie_id': self.movies_df['movie_id'],
            'title': self.movies_df['title'],
            'similarity': similarity_scores
        })
        
        # Exclude movies the user has already rated if requested
        if exclude_seen and user_id in self.ratings_combined['user_id'].values:
            seen_movies = self.ratings_combined[self.ratings_combined['user_id'] == user_id]['movie_id'].values
            recommendations = recommendations[~recommendations['movie_id'].isin(seen_movies)]
        
        # Sort by similarity and return top N
        recommendations = recommendations.sort_values('similarity', ascending=False).head(top_n)
        
        t_end = time.time()
        inference_time = t_end - t_start
        print(f"Recommendations generated in {inference_time:.4f} seconds")
        
        return recommendations, inference_time

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



if __name__ == "__main__":
    movies, users, ratings, watches = read_data()
    content_recommender = ContentBasedRecommender()
    content_recommender.fit(movies, users, ratings, watches)

    # Test the recommender with an existing user
    existing_user_id = ratings['user_id'].iloc[0]  # Get a random user from the dataset
    recommendations, inference_time = content_recommender.get_recommendations(existing_user_id, top_n=20)
    print(f"\nRecommendations for existing user {existing_user_id}:")
    print(recommendations)

    # Test the recommender with a new user (cold start)
    # Create a new user with metadata
    new_user = {
        'age': 30,
        'occupation': 'engineer',
        'gender': 'F'
    }
    # Assign a new user_id that doesn't exist in the dataset
    new_user_id = 999999
    cold_start_recommendations, cold_inference_time = content_recommender.get_recommendations(new_user_id, top_n=20)
    print(f"\nRecommendations for new user (cold start):")
    print(cold_start_recommendations)

    # Print model metrics
    model_size_bytes = content_recommender.get_model_size()
    print(f"\nModel Metrics:")
    print(f"Training Time: {content_recommender.training_time:.2f} seconds")
    print(f"Inference Time (existing user): {inference_time:.4f} seconds")
    print(f"Inference Time (cold start): {cold_inference_time:.4f} seconds")
    print(f"Model Size: {model_size_bytes / (1024*1024):.2f} MB")