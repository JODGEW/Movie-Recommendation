from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate

app = Flask(__name__)

# Load movie details into a DataFrame
try:
    movie_details_df = pd.read_csv('Top_1000_IMDb_movies_New_version.csv')
    print("CSV columns:", movie_details_df.columns.tolist())  # Log the columns in the CSV file
except Exception as e:
    print(f"Error loading CSV file: {e}")

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

TMDB_API_KEY = '61dc118e3630c672f33e35eace3e91b6'

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
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return []

def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('genres', [])
    return []

def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US&append_to_response=credits"
    response = requests.get(url)
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
        detail_response = requests.get(detail_url)
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

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_movies = request.json.get('movies', [])
        if len(user_movies) != 3:
            return jsonify({'error': 'Please select exactly 3 movies.'}), 400

        # Load the ratings data
        ratings_df = pd.read_csv('./ratings_data.csv')

        # Use the Surprise library to handle the dataset and train the model
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

        # Split the data into training and testing sets
        trainset, testset = train_test_split(data, test_size=0.25)

        # Train the SVD algorithm
        algo = SVD()
        algo.fit(trainset)

        # Function to recommend movies for a user
        def recommend_movies(user_movie_ids, num_recommendations=5):
            user_ratings = []
            for movie_id in ratings_df['movieId'].unique():
                if movie_id not in user_movie_ids:
                    user_ratings.append(algo.predict('user_id', movie_id))
            sorted_user_predictions = sorted(user_ratings, key=lambda x: x.est, reverse=True)
            top_recommendations = [int(pred.iid) for pred in sorted_user_predictions[:num_recommendations]]  # Convert to regular int
            return top_recommendations

        recommended_movies = recommend_movies(user_movies)
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
