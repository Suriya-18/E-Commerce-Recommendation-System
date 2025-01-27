import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample product data with user ratings
product_data = {
    'product_id': [1, 2, 3, 4, 5, 6],
    'product_name': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch', 'Camera'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics']
}

user_ratings = {
    'user_id': [1, 1, 1, 2, 2, 2],
    'product_id': [1, 2, 3, 2, 3, 4],
    'rating': [5, 4, 3, 4, 5, 2]
}

# Create DataFrames
products_df = pd.DataFrame(product_data)
ratings_df = pd.DataFrame(user_ratings)

# Create a user-item matrix for collaborative filtering
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating')

# Compute similarity using cosine similarity
similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))

# Function to get recommendations based on a user's rating history
def get_recommendations(user_id, num_recommendations=3):
    user_index = user_id - 1
    similar_users = similarity_matrix[user_index]
    
    # Get products rated by similar users
    similar_user_ratings = user_item_matrix.iloc[similar_users.argsort()[::-1]].fillna(0).iloc[:, :num_recommendations]
    
    recommended_products = products_df.iloc[similar_user_ratings.columns[0]]
    
    print(f"Recommendations for User {user_id}:")
    for index, row in recommended_products.iterrows():
        print(f"{row['product_name']} - {row['category']}")

# Example: Get recommendations for User 1
get_recommendations(1)
