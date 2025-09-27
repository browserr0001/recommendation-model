import pyarrow.parquet as pq
import pandas as pd

def read_data(path_prefix='data/'):
    movies = pq.read_table(f'{path_prefix}meta/movies.parquet').to_pandas()
    users = pq.read_table(f'{path_prefix}meta/users.parquet').to_pandas()
    ratings = pq.read_table(f'{path_prefix}ratings/ratings.parquet').to_pandas()
    watches = pq.read_table(f'{path_prefix}watches/watches.parquet').to_pandas()

    selected_movie_cols = ['id', 'title', 'adult', 'budget', 'genres', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count']
    movies = movies[selected_movie_cols]
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['release_year'] = movies['release_date'].dt.year
    movies.rename(columns={'id': 'movie_id'}, inplace=True)
    

    def get_names(names):
        return [g['name'] for g in names]
    movies['genres'] = movies['genres'].apply(get_names)
    movies['production_companies'] = movies['production_companies'].apply(get_names)
    movies['production_countries'] = movies['production_countries'].apply(get_names)


    numeric_features = ['budget', 'popularity', 'revenue', 'runtime', 
                           'vote_average', 'vote_count']
    for feature in numeric_features:
        movies[feature] = pd.to_numeric(movies[feature], errors='coerce')

    # convert adult to boolean
    movies['adult'] = movies['adult'].astype(bool)
    return movies, users, ratings, watches



if __name__ == "__main__":
    movies, users, ratings, watches = read_data_sample()
    print(movies.shape)
    print(users.shape)
    print(ratings.shape)
    print(watches.shape)