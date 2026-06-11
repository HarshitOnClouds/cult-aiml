import numpy as np
import scipy.sparse as sp
import implicit
from .base import BaseRecommender

class ALSModel(BaseRecommender):
    def __init__(self, factors=100, iterations=20, regularization=0.01):
        """
        Alternating Least Squares (ALS) using `implicit` library.
        Even though Netflix is explicit ratings, ALS can be used by treating 
        ratings as confidence or running explicit ALS.
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, 
            iterations=iterations, 
            regularization=regularization,
            random_state=42
        )
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """
        Fits the implicit ALS model.
        The implicit library expects a sparse matrix of shape (users, items).
        """
        self.user_item_matrix = user_item_matrix
        print(f"Fitting ALS model with factors={self.model.factors}...")
        # implicit uses multi-threading automatically
        self.model.fit(self.user_item_matrix)
        print("ALS model fitted.")

    def predict(self, user_id, movie_id):
        """
        Predict 'rating' or score. 
        Note: ALS trained on implicit feedback outputs scores, not strict 1-5 ratings.
        However, if trained on ratings directly, it can approximate them.
        """
        if self.model.user_factors is None:
            raise ValueError("Model not fitted.")
            
        user_vector = self.model.user_factors[user_id]
        item_vector = self.model.item_factors[movie_id]
        return np.dot(user_vector, item_vector)

    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return Top-N recommendations.
        """
        # implicit has a built-in recommend function
        # Returns tuple of (item_ids, scores)
        ids, scores = self.model.recommend(
            user_id, 
            self.user_item_matrix[user_id], 
            N=n, 
            filter_already_liked_items=exclude_seen
        )
        return ids
