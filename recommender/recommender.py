import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

def load_datasets():
    # Load the datasets
    columns_u_data = ['user_id', 'item_id', 'rating', 'timestamp']
    u_data = pd.read_csv('data/raw/ml-100k/u.data', sep='\t', names=columns_u_data)

    columns_u_user = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    u_user = pd.read_csv('data/raw/ml-100k/u.user', sep='|', names=columns_u_user)

    columns_u_item = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDB_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    u_item = pd.read_csv('data/raw/ml-100k/u.item', sep='|', names=columns_u_item, encoding='latin-1')

    # Merge user data with movies data
    users_and_movies = pd.merge(u_data, u_item[['movie_id', 'title']], left_on='item_id', right_on='movie_id')

    # Create a user-item matrix with ratings
    user_item_movies = users_and_movies.pivot_table(index='user_id', columns='title', values='rating')

    # Fill NaN values with 0 (assuming no rating means a rating of 0)
    user_item_movies = user_item_movies.fillna(0)
    
    return u_data, u_user, u_item, users_and_movies, user_item_movies

def get_content_based_recommendations(userID, user_movies_matrix, top_k=5):
    # Extract the user's ratings
    user_ratings = user_movies_matrix.loc[userID]

    # Create a text representation of the movies, weighted by the user's ratings
    weighted_movies = []
    for title, rating in user_ratings.items():
        if rating > 0:
            weighted_movies.extend([title] * int(rating * 2))  # Weighting by rating

    #Compute the TF-IDF matrix for the movies
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(user_movies_matrix.columns)

    # Transform the weighted movies list into a vector
    user_profile_vector = tfidf_vectorizer.transform([" ".join(weighted_movies)])

    # Compute the cosine similarity
    cosine_similarities = cosine_similarity(user_profile_vector, tfidf_matrix)

    # Sort the movies based on similarity scores
    similarity_scores = list(enumerate(cosine_similarities.flatten()))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top k movie recommendations
    movie_indices = [x[0] for x in similarity_scores[:top_k]]
    recommended_movies = [user_movies_matrix.columns[i] for i in movie_indices]

    return recommended_movies

def get_demographic_based_recommendations(userID, user_movies_matrix, top_k=5):
    # Merge the user demographic data with movie ratings
    merged_data = pd.merge(u_user, user_movies_matrix, left_on='user_id', right_index=True)

    # Group by demographics and calculate average ratings
    grouped_data = merged_data.groupby(['age', 'gender', 'occupation', 'zip_code']).mean()

    # Find the user's demographic group
    user_demographics = u_user.loc[u_user['user_id'] == userID, ['age', 'gender', 'occupation', 'zip_code']]
    if user_demographics.empty:
        return []  # Return an empty list if user's demographic data is not found

    # Extract the user's demographic information
    age, gender, occupation, zip_code = user_demographics.iloc[0]

    # Get the average ratings for this demographic group
    try:
        demographic_ratings = grouped_data.loc[age, gender, occupation, zip_code]
    except KeyError:
        return []  # Return an empty list if the demographic group is not found

    # Sort movies by their average rating within the demographic group
    recommended_movies = demographic_ratings.sort_values(ascending=False).head(top_k).index.tolist()

    return recommended_movies


