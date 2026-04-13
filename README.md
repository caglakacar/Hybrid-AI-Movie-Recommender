<a id="readme-top"></a>

<h1 align="center">Hybrid AI Movie Recommendation System</h1>

<p align="center">
  AI-powered hybrid movie recommendation engine with real-time personalization and scalable deployment.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-F7931E" />
  <img src="https://img.shields.io/badge/NLP-TFIDF%20%7C%20CountVectorizer-4CAF50" />
  <img src="https://img.shields.io/badge/Framework-Flask-000000" />
  <img src="https://img.shields.io/badge/Tracking-MLflow-0194E2" />
  <img src="https://img.shields.io/badge/Cache-Redis-DC382D" />
  <img src="https://img.shields.io/badge/Deployment-Docker-2496ED" />
</p>
<br>

## Table of Contents

1. [Overview](#overview)  
2. [Problem Statement](#problem-statement)  
3. [How It Works](#how-it-works)  
4. [Application Screenshot](#application-screenshot)  
5. [System Architecture](#system-architecture)  
6. [Key Features](#key-features)  
7. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
8. [Usage](#usage)  
9. [Technical Details](#technical-details)  
   - [Dependencies](#dependencies)  
   - [Dataset & Model](#dataset--model)  
   - [Evaluation Results](#evaluation-results)  
   - [Cache Behavior](#cache-behavior)  
   - [MLflow Tracking](#mlflow-tracking)  
   - [Production Considerations](#production-considerations)  
   - [Supporting Files](#supporting-files)  
10. [Folder Structure](#folder-structure)  
11. [License](#license)

<br>

## Overview

This project is a hybrid AI-based movie recommendation system designed to deliver personalized and explainable suggestions using rich movie metadata.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Problem Statement

Traditional recommendation systems often struggle to deliver personalized and explainable results using rich movie metadata. This project addresses this limitation by combining multiple NLP-based models with dynamic weighting and user-driven personalization.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## How It Works

1. Extract movie metadata (genres, cast, director, overview)
2. Create a combined text representation
3. Generate feature vectors using TF-IDF and CountVectorizer
4. Compute similarity using cosine similarity
5. Apply dynamic weighting between models
6. Personalize results using user favorites
7. Cache results for fast retrieval

<br>

The system integrates multiple techniques to improve recommendation quality:

- Hybrid recommendation (TF-IDF + CountVectorizer)
- Dynamic weighting based on movie content
- Personalized recommendations using favorites
- Fast responses with caching
- MLflow experiment tracking
- Docker-based deployment

Designed as a production-ready system that can run seamlessly in both local and containerized environments.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Application Screenshot

<p align="center">
  <img src="static/screenshots/home.png" alt="Home Page" width="900"/>
</p>

<p align="center">
  Main interface showing movie search, recommendations, and favorite actions.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## System Architecture

<p align="center">
  <img src="static/screenshots/architecture.png" alt="System Architecture" width="700"/>
</p>

<p align="center">
  End-to-end pipeline including data processing, hybrid recommendation models, caching layer, and web interface.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Key Features

| **Functionality**            | **Details** |
|-----------------------------|-------------|
| **Hybrid Recommendation**   | Combines TF-IDF and CountVectorizer models for improved accuracy. |
| **Dynamic Weighting**       | Adjusts model importance based on movie metadata richness. |
| **Personalized Results**    | Generates recommendations based on user favorites. |
| **Explainability**          | Provides “why recommended” tags (genre, director, franchise). |
| **Caching System**          | Uses Redis and Flask-Caching for faster responses. |
| **Movie Posters**           | Fetches posters via the TMDB API. |
| **Web Interface**           | Modern responsive UI built with Flask templates. |
| **Autocomplete Search**     | Suggests movie titles dynamically while typing. |
| **MLflow Tracking**         | Tracks model parameters, metrics, and experiments. |
| **Docker Support**          | Containerized deployment with Docker and docker-compose. |
| **Dataset Integration**     | Uses TMDB datasets (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`). |

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- Docker (optional)

<br>

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python train.py
```

Run the app:

```bash
python app.py
```

Or with Docker:

```bash
docker-compose up --build
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Usage

- Search a movie  
- Get recommendations  
- Add/remove favorites  
- Generate personalized recommendations  

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Technical Details

### Dependencies

- Flask  
- pandas  
- numpy  
- scikit-learn  
- requests  
- MLflow  
- Redis  

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

### Dataset & Model

This project uses **The Movies Dataset (TMDB)**:

- `tmdb_5000_movies.csv` → movie metadata  
- `tmdb_5000_credits.csv` → cast & crew  

> **Dataset Source:** [Kaggle - TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download)

#### Similarity Computation

- Text features are combined into a single **content column**
- Includes:
  - genres
  - cast
  - director
  - overview

Models used:

**TF-IDF**
- ngram_range: (1,3)
- max_features: 5000

**CountVectorizer**
- ngram_range: (1,2)
- max_features: 10000

Similarity is computed using **cosine similarity**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

### Evaluation Results

Custom evaluation based on:

- Genre similarity (50%)  
- Director match (30%)  
- Cast similarity (20%)  

Models compared:

- TF-IDF  
- CountVectorizer  
- Hybrid model (best performance)  

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

### Cache Behavior

The application uses Redis cache when Redis is available.  
If Redis is not reachable, it automatically falls back to `SimpleCache`, allowing both local development and Docker-based execution without manual configuration changes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

### MLflow Tracking

Run MLflow to track experiments and model performance:

```bash
mlflow ui --port 5001
```

Open:
```
http://127.0.0.1:5001
```
<br>

<p align="center">
  <img src="static/screenshots/mlflow.png" alt="MLflow Dashboard" width="900"/>
</p>

<p align="center">
  Model experiments, parameters, and metrics tracked using MLflow.
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>



### Production Considerations

- Docker-based containerized deployment  
- Redis caching for performance optimization  
- MLflow for experiment tracking and reproducibility  
- Scalable architecture for future API integration

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

### Supporting Files

- `app.py` → main Flask app  
- `recommender.py` → recommendation logic  
- `train.py` → model training  
- `evaluate.py` → evaluation logic  
- `requirements.txt` → dependencies  

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## Folder Structure

```
project/
│
├── app.py
├── recommender.py
├── train.py
├── evaluate.py
│
├── artifacts/
├── data/
│
├── templates/
│   ├── index.html
│   └── favorites.html
│
├── static/
│   └── screenshots/
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<br>

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>