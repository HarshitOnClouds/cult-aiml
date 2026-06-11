import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

class ItemBasedCF(BaseRecommender):
    def __init__(self, k=50):
        """
        Item-Based Collaborative Filtering.
        
        Args:
            k (int): Number of similar items (neighbors) to consider.
        """
        self.k = k
        self.train_matrix = None
        self.item_similarity = None
        
    def fit(self, train_matrix):
        """
        Computes the item-item similarity matrix.
        Shape: (n_movies, n_movies). For Netflix (17770 movies), this is very manageable in memory.
        """
        self.train_matrix = train_matrix
        print(f"Fitting Item-Based CF (k={self.k}) on matrix of shape {train_matrix.shape}...")
        
        # Transpose to compute similarity between items (columns)
        item_user_matrix = train_matrix.T
        
        # Compute Cosine Similarity between items
        # item_similarity shape: (n_movies, n_movies)
        self.item_similarity = cosine_similarity(item_user_matrix, dense_output=True)
        # Optional: Keep only top K neighbors per item to save memory and speed up inference
        print("Item-Item Similarity matrix computed.")

    def predict(self, user_id, movie_id):
        """
        Predict rating based on similarity-weighted ratings of items the user has seen.
        """
        if self.item_similarity is None:
            raise ValueError("Model is not fitted yet.")
            
        # Get similarities for the target movie
        item_sims = self.item_similarity[movie_id]
        
        # Get all movies rated by this user
        user_ratings = self.train_matrix[user_id].toarray().flatten()
        rated_idx = np.where(user_ratings > 0)[0]
        
        if len(rated_idx) == 0:
             return 3.0
             
        # Filter similarities and ratings
        sims_to_rated = item_sims[rated_idx]
        ratings_from_user = user_ratings[rated_idx]
        
        # Top K similar items rated by user
        if len(sims_to_rated) > self.k:
             top_k_indices = np.argsort(sims_to_rated)[-self.k:]
             sims_to_rated = sims_to_rated[top_k_indices]
             ratings_from_user = ratings_from_user[top_k_indices]
             
        sum_sim = np.sum(np.abs(sims_to_rated))
        if sum_sim == 0:
             return 3.0
             
        pred = np.dot(sims_to_rated, ratings_from_user) / sum_sim
        return pred

    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return Top-N recommendations.
        """
        # Get user's ratings
        user_ratings = self.train_matrix[user_id].toarray().flatten()
        
        # Multiply user ratings by item similarity matrix
        # This computes the raw score for all items simultaneously
        scores = self.item_similarity.dot(user_ratings)
        
        if exclude_seen:
             seen_items = np.where(user_ratings > 0)[0]
             scores[seen_items] = -1
             
        top_n_items = np.argsort(scores)[-n:][::-1]
        return top_n_items
