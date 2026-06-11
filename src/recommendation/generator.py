import numpy as np
import pandas as pd
from collections import defaultdict

class RecommendationGenerator:
    def __init__(self, model, movie_metadata_df, train_matrix=None):
        """
        model: Trained recommendation model (e.g. SVDModel, ALSModel)
        movie_metadata_df: DataFrame with [movie_id, year, title], indexed by encoded movie_index
        train_matrix: Optional, the sparse user-item interaction matrix used for training
        """
        self.model = model
        self.metadata = movie_metadata_df
        self.train_matrix = train_matrix
        self.all_movie_indices = self.metadata.index.tolist()
        
    def _format_movies(self, indices, scores=None):
        """Helper to format movie indices into readable dictionaries"""
        results = []
        for i, idx in enumerate(indices):
            movie_info = self.metadata.loc[idx].to_dict()
            movie_info['movie_index'] = idx
            if scores is not None:
                movie_info['score'] = scores[i]
            results.append(movie_info)
        return results

    def get_top_k(self, user_id, k=10, exclude_seen=True):
        """
        Return top-K movies with scores and metadata.
        """
        seen_movies = set()
        if exclude_seen and self.train_matrix is not None:
             # get non-zero entries for user row
             seen_movies = set(self.train_matrix[user_id].indices)
             
        # Call model's recommend function
        # We assume models have a recommend() signature handling this, or we handle it uniformly.
        # Since different models have different signatures in our base code, let's adapt:
        if hasattr(self.model, 'recommend'):
             try:
                 top_indices = self.model.recommend(user_id, n=k, exclude_seen=exclude_seen)
             except TypeError:
                 # If SVDModel, signature is slightly different
                 top_indices = self.model.recommend(user_id, self.all_movie_indices, n=k, 
                                                   exclude_seen=exclude_seen, seen_movies=seen_movies)
        else:
             raise NotImplementedError("Model does not implement recommend()")
             
        # We also want scores if the model can provide them, but just indices is fine for now
        scores = [self.model.predict(user_id, idx) for idx in top_indices]
        
        return self._format_movies(top_indices, scores)

    def get_similar_movies(self, movie_id, k=10):
        """
        Return K most similar movies using item factors if available.
        """
        if hasattr(self.model, 'item_similarity') and self.model.item_similarity is not None:
            sims = self.model.item_similarity[movie_id]
            top_k_indices = np.argsort(sims)[-(k+1):-1][::-1] # Exclude self
            scores = sims[top_k_indices]
            return self._format_movies(top_k_indices, scores)
            
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'item_factors'):
            # ALS model
            item_factors = self.model.model.item_factors
            target_vec = item_factors[movie_id]
            # Compute dot product (un-normalized cosine)
            sims = np.dot(item_factors, target_vec)
            top_k_indices = np.argsort(sims)[-(k+1):-1][::-1]
            scores = sims[top_k_indices]
            return self._format_movies(top_k_indices, scores)
            
        else:
             raise NotImplementedError("Model does not support similar items extraction directly.")

    def explain_recommendation(self, user_id, movie_id):
        """
        Explainability: 'You might like X because you enjoyed Y and Z...'
        Simple implementation based on item similarity if available.
        """
        if not hasattr(self.model, 'item_similarity'):
            return "Explanation not available for this model type (requires Item-Item similarity)."
            
        if self.train_matrix is None:
            return "Training matrix not provided."
            
        # Get movies user liked (rating >= 4)
        user_ratings = self.train_matrix[user_id].toarray().flatten()
        liked_indices = np.where(user_ratings >= 4.0)[0]
        
        if len(liked_indices) == 0:
            return "You might like this based on general popularity (no strong personal history found)."
            
        # Get similarities between target movie and liked movies
        target_sims = self.model.item_similarity[movie_id]
        sims_with_liked = target_sims[liked_indices]
        
        # Top 2 most similar liked movies
        if len(sims_with_liked) > 0:
            top_2 = np.argsort(sims_with_liked)[-2:][::-1]
            top_liked_indices = liked_indices[top_2]
            
            movie_names = [self.metadata.loc[idx, 'title'] for idx in top_liked_indices]
            target_name = self.metadata.loc[movie_id, 'title']
            
            if len(movie_names) == 2:
                return f"You might like '{target_name}' because you enjoyed '{movie_names[0]}' and '{movie_names[1]}'."
            elif len(movie_names) == 1:
                return f"You might like '{target_name}' because you enjoyed '{movie_names[0]}'."
                
        return "Recommended based on your latent preferences."

    def batch_recommend(self, user_ids, k=10):
        """
        Efficient batch recommendation for multiple users.
        """
        results = {}
        for uid in user_ids:
            results[uid] = self.get_top_k(uid, k=k)
        return results
