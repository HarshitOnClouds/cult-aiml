import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

class UserBasedCF(BaseRecommender):
    def __init__(self, k=50):
        """
        User-Based Collaborative Filtering.
        
        Args:
            k (int): Number of similar users (neighbors) to consider.
        """
        self.k = k
        self.train_matrix = None
        self.user_similarity = None
        
    def fit(self, train_matrix):
        """
        Computes the user-user similarity matrix.
        Note: For 100M dataset, computing full user-user similarity (480k x 480k) 
        is OOM. This implementation works for the subsets, or computes similarity on-the-fly.
        """
        self.train_matrix = train_matrix
        print(f"Fitting User-Based CF (k={self.k}) on matrix of shape {train_matrix.shape}...")
        
        # In a real large-scale scenario, we would use FAISS or Annoy for approximate nearest neighbors.
        # For the Tiny/Medium subsets, we will compute similarity directly.
        # If the matrix is too large (e.g. users > 50,000), we'll print a warning.
        n_users = train_matrix.shape[0]
        if n_users > 50000:
            print("Warning: Dense User-User similarity matrix will be very large. Consider TruncatedSVD first or use FAISS.")
            
        # Compute Cosine Similarity between users
        # user_similarity shape: (n_users, n_users)
        self.user_similarity = cosine_similarity(train_matrix, dense_output=False)
        print("Similarity matrix computed.")

    def predict(self, user_id, movie_id):
        """
        Predict rating using weighted average of k-nearest neighbors' ratings.
        """
        if self.train_matrix is None:
            raise ValueError("Model is not fitted yet.")
            
        # Get similarities for the target user
        if sp.issparse(self.user_similarity):
            user_sims = self.user_similarity[user_id].toarray().flatten()
        else:
            user_sims = self.user_similarity[user_id]
            
        # Get all users who rated this movie
        # To do this fast, we look at the column of the train_matrix (item-user matrix would be faster here)
        movie_col = self.train_matrix[:, movie_id].toarray().flatten()
        raters_idx = np.where(movie_col > 0)[0]
        
        if len(raters_idx) == 0:
            return 3.0 # Fallback to global average or neutral
            
        # Filter similarities and ratings
        sims_to_raters = user_sims[raters_idx]
        ratings_from_raters = movie_col[raters_idx]
        
        # Select Top-K similar users
        if len(sims_to_raters) > self.k:
            top_k_indices = np.argsort(sims_to_raters)[-self.k:]
            sims_to_raters = sims_to_raters[top_k_indices]
            ratings_from_raters = ratings_from_raters[top_k_indices]
            
        # Weighted average
        sum_sim = np.sum(np.abs(sims_to_raters))
        if sum_sim == 0:
            return 3.0
            
        pred = np.dot(sims_to_raters, ratings_from_raters) / sum_sim
        return pred

    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return Top-N recommendations.
        """
        # We can score all items at once by taking the weighted sum of all items rated by top K neighbors
        if sp.issparse(self.user_similarity):
            user_sims = self.user_similarity[user_id].toarray().flatten()
        else:
            user_sims = self.user_similarity[user_id]
            
        # Top K neighbors
        top_k_neighbors = np.argsort(user_sims)[-(self.k+1):-1] # Exclude self (which is highest)
        
        # Neighbor similarities
        k_sims = user_sims[top_k_neighbors]
        
        # Get neighbors' ratings (shape: K x num_items)
        k_ratings = self.train_matrix[top_k_neighbors].toarray()
        
        # Weighted sum of ratings
        # Shape: (num_items,)
        item_scores = np.dot(k_sims, k_ratings) / (np.sum(np.abs(k_sims)) + 1e-9)
        
        if exclude_seen:
            seen_items = self.train_matrix[user_id].indices
            item_scores[seen_items] = -1 # Make seen items score very low
            
        # Get top N items
        top_n_items = np.argsort(item_scores)[-n:][::-1]
        return top_n_items
