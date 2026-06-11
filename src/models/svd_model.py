import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from .base import BaseRecommender

class SVDModel(BaseRecommender):
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Matrix Factorization using SVD (scikit-surprise).
        """
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, 
                         lr_all=lr_all, reg_all=reg_all, random_state=42)
        self.trainset = None
        self.user_mapping = {} # surprise inner id -> original/encoded id
        self.item_mapping = {}
        
    def fit(self, train_df):
        """
        Fits the Surprise SVD model.
        Note: Surprise expects a DataFrame, not a sparse matrix, so we pass
        train_df (DataFrame) with columns: [user_index, movie_index, rating]
        """
        print(f"Fitting SVD model on {len(train_df)} ratings...")
        reader = Reader(rating_scale=(1, 5))
        # Surprise needs the columns in exactly this order
        data = Dataset.load_from_df(train_df[['user_index', 'movie_index', 'rating']], reader)
        self.trainset = data.build_full_trainset()
        
        self.model.fit(self.trainset)
        print("SVD model fitted.")

    def predict(self, user_id, movie_id):
        """
        Predict rating.
        """
        # surprise predict handles unknown users/items gracefully
        pred = self.model.predict(uid=user_id, iid=movie_id)
        return pred.est

    def recommend(self, user_id, all_movie_ids, n=10, exclude_seen=True, seen_movies=None):
        """
        Return Top-N recommendations.
        
        Args:
            user_id: Encoded user index.
            all_movie_ids: List of all valid encoded movie indices.
            n: Number of recs.
            exclude_seen: Whether to exclude movies already seen.
            seen_movies: Set of movies the user has already rated (required if exclude_seen=True)
        """
        candidate_movies = all_movie_ids
        if exclude_seen and seen_movies is not None:
            candidate_movies = [m for m in all_movie_ids if m not in seen_movies]
            
        # Predict score for all candidate movies
        predictions = [self.model.predict(uid=user_id, iid=m) for m in candidate_movies]
        
        # Sort by estimated rating
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        top_n = [pred.iid for pred in predictions[:n]]
        return top_n
