import pandas as pd
import numpy as np

# Load the dataset
movies_df = pd.read_csv('./Top_1000_IMDb_movies_New_version.csv')

# Simulate user ratings
num_users = 100
num_movies = len(movies_df)
np.random.seed(42)

# Generate user IDs, movie IDs, and ratings
user_ids = np.random.randint(1, num_users + 1, size=5*num_movies)
movie_ids = np.random.randint(0, num_movies, size=5*num_movies)
ratings = np.random.randint(1, 6, size=5*num_movies)

ratings_df = pd.DataFrame({
    'userId': user_ids,
    'movieId': movie_ids,
    'rating': ratings
})

# Merge simulated ratings with movie details
ratings_df = ratings_df.merge(movies_df[['Unnamed: 0', 'Movie Name']], left_on='movieId', right_on='Unnamed: 0', how='left')
ratings_df = ratings_df[['userId', 'movieId', 'rating', 'Movie Name']]

# Aggregate duplicate entries by averaging the ratings
ratings_df = ratings_df.groupby(['userId', 'movieId']).agg({'rating': 'mean'}).reset_index()

# Save the ratings data to a CSV file
ratings_df.to_csv('./data/ratings_data.csv', index=False)

# Display the first few rows
print(ratings_df.head())
