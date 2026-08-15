"""Build ratings + genre data from real MovieLens ratings.

Matches MovieLens movies to the Top-1000 IMDb movies by normalized title
(+ year when ambiguous), then maps each rating onto the local movie ids
used by the app ('Unnamed: 0' index of the Top-1000 CSV).

Works with any MovieLens release (ml-latest-small for the committed CSVs,
ml-25m for training runs).

Usage:
    python python/data.py /path/to/ml-latest-small

Download datasets from https://grouplens.org/datasets/movielens/
"""
import os
import re
import sys

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')

ARTICLES = ('the', 'a', 'an', 'la', 'le', 'les', 'el', 'il')


def normalize_title(title):
    t = str(title).lower().strip()
    t = re.sub(r'\([^)]*\)', '', t)  # drop "(1995)" / alternate titles
    t = t.strip()
    m = re.match(r'^(.*),\s*(%s)$' % '|'.join(ARTICLES), t)  # "shawshank redemption, the"
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = re.sub(r'[^a-z0-9]+', ' ', t).strip()
    return t


def extract_year(ml_title):
    m = re.search(r'\((\d{4})\)\s*$', str(ml_title))
    return int(m.group(1)) if m else None


def match_movies(ml_dir):
    """Match MovieLens movies to Top-1000 movies.

    Returns (ml_to_local, ml_movies): a MovieLens movieId -> local id dict
    and the MovieLens movies DataFrame."""
    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'Top_1000_IMDb_movies_New_version.csv'))
    ml_movies = pd.read_csv(os.path.join(ml_dir, 'movies.csv'))

    movies_df['norm_title'] = movies_df['Movie Name'].apply(normalize_title)
    ml_movies['norm_title'] = ml_movies['title'].apply(normalize_title)
    ml_movies['year'] = ml_movies['title'].apply(extract_year)

    candidates = {}
    for _, row in movies_df.iterrows():
        candidates.setdefault(row['norm_title'], []).append(
            (row['Unnamed: 0'], row['Year of Release'])
        )

    ml_to_local = {}
    for _, row in ml_movies.iterrows():
        matches = candidates.get(row['norm_title'])
        if not matches:
            continue
        if len(matches) == 1:
            ml_to_local[row['movieId']] = matches[0][0]
        else:
            # Disambiguate by year (allow +/-1 for release date discrepancies)
            for local_id, year in matches:
                if row['year'] is not None and abs(int(year) - row['year']) <= 1:
                    ml_to_local[row['movieId']] = local_id
                    break
    return ml_to_local, ml_movies


def build_ratings(ml_dir, ml_to_local=None):
    """Real ratings mapped onto local movie ids: userId, movieId, rating."""
    if ml_to_local is None:
        ml_to_local, _ = match_movies(ml_dir)
    ml_ratings = pd.read_csv(os.path.join(ml_dir, 'ratings.csv'),
                             usecols=['userId', 'movieId', 'rating'])
    ratings = ml_ratings[ml_ratings['movieId'].isin(ml_to_local)].copy()
    ratings['movieId'] = ratings['movieId'].map(ml_to_local)
    ratings = ratings.groupby(['userId', 'movieId']).agg({'rating': 'mean'}).reset_index()

    print(f"Matched {len(ml_to_local)} of the Top-1000 movies to MovieLens")
    print(f"Kept {len(ratings)} real ratings from {ratings['userId'].nunique()} users")
    return ratings[['userId', 'movieId', 'rating']]


def build_genres(ml_dir, ml_to_local=None, ml_movies=None):
    """Genre words per local movie id: movieId, genres ('action sci-fi ...')."""
    if ml_to_local is None or ml_movies is None:
        ml_to_local, ml_movies = match_movies(ml_dir)
    rows = []
    for _, row in ml_movies[ml_movies['movieId'].isin(ml_to_local)].iterrows():
        genres = str(row['genres'])
        if genres == '(no genres listed)':
            genres = ''
        rows.append({
            'movieId': ml_to_local[row['movieId']],
            'genres': genres.lower().replace('|', ' '),
        })
    return pd.DataFrame(rows).drop_duplicates('movieId')


if __name__ == '__main__':
    ml_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, 'ml-latest-small')
    ml_to_local, ml_movies = match_movies(ml_dir)

    ratings_df = build_ratings(ml_dir, ml_to_local)
    ratings_path = os.path.join(DATA_DIR, 'ratings_data.csv')
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Saved to {ratings_path}")

    genres_df = build_genres(ml_dir, ml_to_local, ml_movies)
    genres_path = os.path.join(DATA_DIR, 'movie_genres.csv')
    genres_df.to_csv(genres_path, index=False)
    print(f"Saved {len(genres_df)} genre rows to {genres_path}")
