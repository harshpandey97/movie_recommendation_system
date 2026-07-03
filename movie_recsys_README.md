# Hybrid Movie Recommendation System (TMDB 5000 Dataset)

A hybrid movie recommendation system built with Python, Pandas, and Scikit-learn, using the
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.

## Project Description

Given a movie title, this project recommends similar movies using a **hybrid** approach:

1. **Content-based filtering** - TF-IDF vectorization + cosine similarity over each movie's genres,
   overview, top 5 cast members, director, and keywords.
2. **Weighted rating (IMDB-style Bayesian average)** - re-ranks the content-similar candidates by
   a rating that accounts for vote count, so a movie with 5 votes at 9.0 doesn't outrank one with
   5,000 votes at 8.4.

The two signals are blended into a single `hybrid_score` to produce the final ranked recommendation list.

> **Note on collaborative filtering:** true collaborative filtering needs a user-item ratings
> history (many users, many movies, who rated what). This dataset only has per-movie aggregate
> stats (not individual user ratings), so the "hybrid" here blends content similarity with a
> popularity/rating-weighted signal instead - the standard practical approach for catalog-only
> data. A true user-based CF layer is a natural next extension using a dataset like MovieLens,
> which includes individual user ratings.

## Dataset

- Source: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)
- `tmdb_5000_movies.csv` - 4,803 movies with budget, genres, keywords, overview, popularity, ratings, etc.
- `tmdb_5000_credits.csv` - cast and crew for each movie

## What This Project Does

- Loads and merges the movies and credits files on movie ID
- Inspects and handles missing values column by column (dropped `homepage`/`tagline`, dropped rows
  missing `overview`/`release_date`, median-filled `runtime`)
- Parses the nested JSON-like `genres`, `keywords`, `cast`, and `crew` columns into clean lists
- Engineers a combined "tags" feature (genres + overview + cast + director + keywords)
- Builds a TF-IDF matrix and cosine similarity matrix for content-based filtering
- Computes an IMDB-style weighted rating for every movie
- Combines both into a `recommend(title)` function returning a hybrid-ranked recommendation list
- Visualizes: top genres, popularity vs. rating, top-rated movies, rating distribution, movies
  released per year
- Exports the processed dataset to `tmdb_movies_processed.csv`

## Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn (TF-IDF, cosine similarity)
- Matplotlib / Seaborn (visualizations)
- Jupyter Notebook

## Sample Output

```
recommend('The Avengers', top_n=10)

                                  title  vote_average  vote_count  similarity  hybrid_score
0              Avengers: Age of Ultron           7.3        6767    0.408        ...
1  Captain America: The Winter Soldier           7.6        5764    0.219        ...
2              Guardians of the Galaxy           7.9        9742    0.136        ...
3                             Iron Man           7.4        8776    0.183        ...
...
```

## How to Run

1. Place `movie_recommendation_system.ipynb`, `tmdb_5000_movies.csv`, and `tmdb_5000_credits.csv`
   in the same folder.
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open the notebook in Jupyter and run all cells.
4. Use `recommend('Movie Title')` at the bottom to get recommendations for any movie in the dataset.

## Files

| File | Description |
|---|---|
| `movie_recommendation_system.ipynb` | Main notebook - cleaning, feature engineering, hybrid recommender, visualizations (already executed) |
| `tmdb_5000_movies.csv` | Raw movies dataset |
| `tmdb_5000_credits.csv` | Raw cast/crew dataset |
| `tmdb_movies_processed.csv` | Cleaned, processed output |
| `viz_top_genres.png` | Top 10 genres in the dataset |
| `viz_popularity_vs_rating.png` | Popularity vs. average rating scatter plot |
| `viz_top_rated.png` | Top 10 movies by weighted rating |
| `viz_rating_distribution.png` | Distribution of average ratings |
| `viz_movies_per_year.png` | Movies released per year (1980 onward) |

## What You'll Learn

- Merging and cleaning two related real-world datasets
- Parsing nested JSON-like string columns (a common messy-data pattern)
- Building a content-based recommender from scratch with TF-IDF and cosine similarity
- Why raw average ratings are misleading, and how a Bayesian weighted rating fixes that
- Combining two different signals into a single hybrid ranking score
