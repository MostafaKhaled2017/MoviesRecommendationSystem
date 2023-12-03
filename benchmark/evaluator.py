import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from recommender.recommender import get_all_recommendations, u_data, user_item_movies

test_data = pd.read_csv('benchmark/test_data.csv', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Function to calculate MSE and RMSE
def evaluate_recommender(recommender_func, user_movies_matrix, test_data):
    predicted_ratings = []
    actual_ratings = []

    for _, row in test_data.iterrows():
        user_id, item_id, actual_rating, _ = row
        recommended_movies = recommender_func(user_id, user_movies_matrix)

        # Check if the recommended movie is in the test set
        if item_id in recommended_movies:
            predicted_ratings.append(1)  # Recommended
        else:
            predicted_ratings.append(0)  # Not recommended

        actual_ratings.append(actual_rating)

    mse = mean_squared_error(actual_ratings, predicted_ratings)
    rmse = sqrt(mse)

    return mse, rmse

# Evaluate the recommender system
mse, rmse = evaluate_recommender(get_all_recommendations, user_item_movies, test_data)
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
