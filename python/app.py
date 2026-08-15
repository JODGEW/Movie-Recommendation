from flask import Flask, jsonify, render_template, request
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from surprise import Dataset, Reader, SVD

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'transformer.pt')

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static')
)

# Load movie details into a DataFrame
NAME_TO_ID = {}
try:
    movie_details_df = pd.read_csv(os.path.join(DATA_DIR, 'Top_1000_IMDb_movies_New_version.csv'))
    print("CSV columns:", movie_details_df.columns.tolist())  # Log the columns in the CSV file
    for _, row in movie_details_df.iterrows():
        NAME_TO_ID.setdefault(str(row['Movie Name']), int(row['Unnamed: 0']))
except Exception as e:
    print(f"Error loading CSV file: {e}")


def load_transformer_recommender():
    """Load the trained Transformer checkpoint; returns None if unavailable."""
    if not os.path.exists(MODEL_PATH):
        print("No transformer checkpoint found, /recommend will use SVD fallback")
        return None
    try:
        import numpy as np
        import torch
        from sklearn.preprocessing import LabelEncoder
        from model import Transformer, get_recommendations

        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        model = Transformer(
            src_vocab_size=ckpt['vocab_size'], tgt_vocab_size=ckpt['vocab_size'],
            max_seq_len=ckpt['max_seq_len'], **ckpt['hparams'],
        )
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(ckpt['label_classes'])
        print("Transformer recommender loaded")
        return {
            'model': model,
            'vocab': ckpt['vocab'],
            'label_encoder': label_encoder,
            'max_seq_len': ckpt['max_seq_len'],
            'recommend': get_recommendations,
        }
    except Exception as e:
        print(f"Could not load transformer checkpoint: {e}")
        return None


transformer_recommender = load_transformer_recommender()

_svd_cache = None

def get_svd_recommender():
    """Train the SVD fallback once and cache it."""
    global _svd_cache
    if _svd_cache is None:
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings_data.csv'))
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        algo = SVD()
        algo.fit(data.build_full_trainset())
        _svd_cache = (ratings_df, algo)
    return _svd_cache

def get_imdb_movie_details(movie_id):
    try:
        print(f"Fetching details for IMDb movie ID: {movie_id}")
        movie = movie_details_df[movie_details_df['Unnamed: 0'] == movie_id]
        if not movie.empty:
            print(f"Movie found: {movie}")
            # Remove commas from 'Votes' and 'Gross' columns and convert to appropriate types
            votes_str = movie.iloc[0]['Votes']
            votes = int(votes_str.replace(',', '')) if pd.notna(votes_str) else 0

            gross_str = movie.iloc[0]['Gross']
            if isinstance(gross_str, str):
                gross = float(gross_str.replace(',', ''))
            else:
                gross = gross_str if pd.notna(gross_str) else 0.0

            movie_details = {
                'movieId': int(movie_id),
                'title': movie.iloc[0]['Movie Name'],
                'year': int(movie.iloc[0]['Year of Release']) if pd.notna(movie.iloc[0]['Year of Release']) else 'N/A',
                'watch_time': int(movie.iloc[0]['Watch Time']) if pd.notna(movie.iloc[0]['Watch Time']) else 'N/A',
                'rating': float(movie.iloc[0]['Movie Rating']) if pd.notna(movie.iloc[0]['Movie Rating']) else 'N/A',
                'metascore': int(movie.iloc[0]['Metascore of movie']) if pd.notna(movie.iloc[0]['Metascore of movie']) else 'N/A',
                'gross': gross,
                'votes': votes,
                'description': movie.iloc[0]['Description'] if pd.notna(movie.iloc[0]['Description']) else 'N/A'
            }
            return movie_details
        else:
            print(f"No movie found with ID: {movie_id}")
    except KeyError as e:
        print(f"KeyError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    return None

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY is not set. Copy .env.example to .env and fill in your key.")

def fetch_movies(page=1, genre=None, year=None, language=None):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&language=en-US&sort_by=popularity.desc&page={page}"
    if genre:
        url += f"&with_genres={genre}"
    if year:
        if '-' in year:
            start_year, end_year = year.split('-')
            url += f"&primary_release_date.gte={start_year}-01-01&primary_release_date.lte={end_year}-12-31"
        elif year == '90s':
            url += f"&primary_release_date.gte=1990-01-01&primary_release_date.lte=1999-12-31"
        elif year == '80s':
            url += f"&primary_release_date.gte=1980-01-01&primary_release_date.lte=1989-12-31"
        elif year == 'before_80s':
            url += f"&primary_release_date.lte=1979-12-31"
        else:
            url += f"&primary_release_year={year}"
    if language:
        url += f"&with_original_language={language}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return []

def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        return response.json().get('genres', [])
    return []

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=credits"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None

@app.route('/')
def index():
    genres = fetch_genres()
    return render_template('index.html', genres=genres)

@app.route('/api/movies')
def get_movies():
    page = request.args.get('page', 1, type=int)
    genre = request.args.get('genre')
    year = request.args.get('year')
    language = request.args.get('language')
    movies = fetch_movies(page, genre, year, language)
    detailed_movies = []
    for movie_data in movies:
        movie_id = movie_data['id']
        detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=credits"
        detail_response = requests.get(detail_url, timeout=10)
        if detail_response.status_code == 200:
            detail_data = detail_response.json()
            detailed_movies.append({
                'id': movie_id,
                'name': detail_data.get('title', 'N/A'),
                'poster': f"https://image.tmdb.org/t/p/w500{detail_data.get('poster_path', '')}",
                'cast': ', '.join([cast['name'] for cast in detail_data.get('credits', {}).get('cast', [])[:3]]),
                'type': ', '.join([genre['name'] for genre in detail_data.get('genres', [])]),
                'year': detail_data.get('release_date', '')[:4] if detail_data.get('release_date') else 'N/A',
                'overview': detail_data.get('overview', 'No overview available'),
                'director': ', '.join([crew['name'] for crew in detail_data.get('credits', {}).get('crew', []) if crew['job'] == 'Director'])
            })
    return jsonify(detailed_movies)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    movie = fetch_movie_details(movie_id)
    if movie:
        return render_template('movie_detail.html', movie=movie)
    else:
        return "Movie not found", 404

# TMDB genre names that differ from the MovieLens genre vocabulary the
# model was trained on
GENRE_SYNONYMS = {'science fiction': 'sci-fi', 'family': 'children', 'music': 'musical'}

def recommend_with_transformer(tmdb_movie_ids, num_recommendations=5):
    """Look up the selected movies' titles and genres on TMDB and ask the
    trained Transformer for similar Top-1000 movies. Returns local movie ids."""
    titles, texts = [], []
    for tmdb_id in tmdb_movie_ids:
        details = fetch_movie_details(tmdb_id)
        if not details or not details.get('title'):
            return None
        titles.append(details['title'])
        genres = [GENRE_SYNONYMS.get(g['name'].lower(), g['name'].lower())
                  for g in details.get('genres', [])]
        texts.append(f"{details['title']} {' '.join(genres)}".strip())

    r = transformer_recommender
    known_words = [w for t in texts for w in t.lower().split() if w in r['vocab']]
    if not known_words:
        return None
    names = r['recommend'](
        texts, r['vocab'], r['model'], r['label_encoder'],
        r['max_seq_len'], device='cpu', top_k=25,
    )
    selected = {t.lower() for t in titles}
    recommendations = []
    for name in names:
        movie_id = NAME_TO_ID.get(name)
        if movie_id is None or name.lower() in selected or movie_id in recommendations:
            continue
        recommendations.append(movie_id)
        if len(recommendations) == num_recommendations:
            break
    return recommendations or None


def recommend_with_svd(user_movie_ids, num_recommendations=5):
    """Fallback: rank the rated movies by predicted score for a generic user."""
    ratings_df, algo = get_svd_recommender()
    predictions = [
        algo.predict('user_id', movie_id)
        for movie_id in ratings_df['movieId'].unique()
        if movie_id not in user_movie_ids
    ]
    predictions.sort(key=lambda p: p.est, reverse=True)
    return [int(p.iid) for p in predictions[:num_recommendations]]


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_movies = request.json.get('movies', [])
        if len(user_movies) != 3:
            return jsonify({'error': 'Please select exactly 3 movies.'}), 400

        recommended_movies = None
        if transformer_recommender:
            recommended_movies = recommend_with_transformer(user_movies)
        if not recommended_movies:
            recommended_movies = recommend_with_svd(user_movies)

        print(f"Recommended movies: {recommended_movies}")  # Log recommendations
        return jsonify(recommended_movies)
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/imdb_movie/<int:movie_id>')
def imdb_movie_detail(movie_id):
    print(f"Fetching details for IMDb movie ID: {movie_id}")
    movie = get_imdb_movie_details(movie_id)
    if movie:
        return jsonify(movie)
    else:
        print(f"Movie with ID {movie_id} not found in IMDb data")
        return jsonify({'error': 'Movie not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
