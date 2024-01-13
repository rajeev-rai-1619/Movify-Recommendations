import pandas as pd
import streamlit as st

# Load the data
rating = pd.read_csv("ratings.csv")
movie = pd.read_csv("movies.csv")
rating_new = pd.merge(rating, movie)

# Pivot the data to create the user-item matrix
user_item_matrix = rating_new.pivot_table(index=['userId'], columns=['title'], values='rating')
user_item_matrix = user_item_matrix.dropna(thresh=10, axis=1)
user_item_matrix = user_item_matrix.fillna(0)

# Calculate Pearson Similarity
user_similarity = user_item_matrix.corr(method='pearson')


# Function to get similar movies
def get_similar_movies(movie_name, user_rating):
    similar_score = user_similarity[movie_name] * (user_rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score


# Streamlit app
st.title("Movie Recommendation System")

# User input for movie ratings
selected_movies = st.multiselect("Select movies you have watched and rate them:", user_item_matrix.columns)
user_ratings = {}
for movie_title in selected_movies:
    user_ratings[movie_title] = st.slider(f"Rate {movie_title}", 1.0, 5.0, 3.0, 1.0)
# Display user ratings
st.write("### Your Ratings:")
st.write(user_ratings)

# Generate movie recommendations
similar_movies = pd.DataFrame()
for movie_title, rating in user_ratings.items():
    similar_movies = pd.concat([similar_movies, get_similar_movies(movie_title, rating)], axis=1)

# Sum up the scores for each movie+
recommended_movies = similar_movies.sum(axis=1).sort_values(ascending=False)

# Remove movies already rated by the user
rated_movies = pd.Series(user_ratings)
recommended_movies = recommended_movies[~recommended_movies.index.isin(rated_movies.index)]

# Display recommended movies
st.write("### Recommended Movies:")
st.write(recommended_movies.head(10).index.tolist())
