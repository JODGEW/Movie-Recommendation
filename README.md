# Movie-Recommendation
A web app for a movie recommendation system that fetches detailed movie information from TMDb for the front-end, ensuring up-to-date and rich metadata. For the backend, IMDb data is utilized to train a custom transformer deep learning model, leveraging user ratings and reviews to provide highly accurate and personalized recommendations. This approach combines real-time data fetching with advanced deep learning techniques to enhance user experience and movie discovery.

## Main Page
<p align="center">
  <img src="images/page.png" alt="Main Page" width="800">
</p>

## Recommendations from Top Five Guesses Based on User-Selected Movies
<p align="center">
  <img src="images/rec.png" alt="Recommendation" width="600">
</p>

<p align="center">
  <img src="images/rec1.png" alt="Recommendation 1" width="600">
</p>

<p align="center">
  <img src="images/rec2.png" alt="Recommendation 2" width="600">
</p>

## Steps to Run:

1. **Navigate to the directory that you clone project at in the terminal**:
   ```sh
   cd /path/to/your/Movie-Recommendation

2. **Install dependencies using**:
    ```sh
    pip install -r requirements.txt

3. **Run the Flask application**:
    ```sh
    python app.py

4. Open your web browser and go to `http://127.0.0.1:5000/`.


## Reference:
[Attention is All You Need](https://arxiv.org/abs/1706.03762)

[IMDb Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset?)

[TMDB API](https://developer.themoviedb.org/reference/intro/getting-started)