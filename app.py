from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import requests

app = Flask(__name__)
app.secret_key = 'movie_recommender'

TMDB_API_KEY = "c43e5de6820ef7a74c4ba82a76de370b"

def get_poster_url(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url)
        data = response.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path", None)
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return "https://via.placeholder.com/150x220?text=No+Image"


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

def convert(obj):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(obj)][:3])
    except:
        return ""

def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
        return ""
    except:
        return ""


movies['genres'] = movies['genres'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['cast'] = movies['cast'].fillna('')
movies['crew'] = movies['crew'].fillna('')

movies['combined'] = (
    movies['genres'] + ' ' + movies['genres'] + ' ' + movies['genres'] + ' ' +
    movies['cast'] + ' ' + movies['cast'] + ' ' + movies['cast'] + ' ' +
    movies['crew'] + ' ' + movies['crew'] + ' ' +
    movies['overview']
)


vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    min_df=1,
    max_features=10000
)
matrix = vectorizer.fit_transform(movies['combined'])
similarity = cosine_similarity(matrix)


def get_recommendations(title):
    title = title.strip().lower()
    all_titles = movies['title'].str.lower()
    matches = movies[all_titles.str.contains(title)]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = [s for s in sorted_scores if s[0] != idx][:8]

    results = []
    favorites = session.get("favorites", [])
    for i in sorted_scores:
        movie = movies.iloc[i[0]]
        results.append({
            "title": movie['title'],
            "genre": movie['genres'],
            "description": movie['overview'],
            "director": movie['crew'],
            "poster": get_poster_url(movie["title"]),
            "similarity": round(i[1] * 100, 1),
            "trailer": f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer",
            "is_favorite": movie['title'].lower() in [f.lower() for f in favorites]
        })

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query = ""
    if request.method == "POST":
        query = request.form["movie"]
        recommendations = get_recommendations(query)
    all_titles = sorted(movies["title"].unique().tolist())
    return render_template("index.html", recommendations=recommendations, query=query, all_titles=all_titles)


@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("q", "").lower()
    suggestions = []
    if query:
        matched_titles = movies[movies["title"].str.lower().str.contains(query, na=False)]["title"].tolist()
        matched_titles = sorted(matched_titles, key=lambda x: x.lower() != query)
        suggestions = matched_titles[:8]
    return jsonify(suggestions)


@app.route("/add_favorite", methods=["POST"])
def add_favorite():
    title = request.json.get("title")
    if not title:
        return jsonify({"status": "error", "message": "Title is required"}), 400

    favorites = session.get("favorites", [])
    if title.lower() not in [f.lower() for f in favorites]:
        favorites.append(title)
        session["favorites"] = favorites

    return jsonify({"status": "success"})


@app.route("/favorites")
def favorites():
    favorite_titles = session.get("favorites", [])
    favorite_movies = []
    for title in favorite_titles:
        movie_data = movies[movies['title'].str.lower() == title.lower()]
        if not movie_data.empty:
            movie = movie_data.iloc[0]
            favorite_movies.append({
                "title": movie['title'],
                "genre": movie['genres'],
                "description": movie['overview'],
                "director": movie['crew'],
                "poster": get_poster_url(movie['title']),
                "trailer": f"https://www.youtube.com/results?search_query={movie['title'].replace(' ', '+')}+trailer"
            })
    return render_template("favorites.html", favorites=favorite_movies)


@app.route("/remove_favorite", methods=["POST"])
def remove_favorite():
    title = request.json.get("title")
    if not title:
        return jsonify({"status": "error", "message": "Title is required"}), 400

    favorites = session.get("favorites", [])
    for f in favorites:
        if f.lower() == title.lower():
            favorites.remove(f)
            session["favorites"] = favorites
            break

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)