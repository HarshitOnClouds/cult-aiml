from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """
    Base interface for all recommendation models.
    Ensures consistency across different model implementations.
    """
    
    @abstractmethod
    def fit(self, train_matrix):
        """
        Train the model on the provided user-item matrix.
        
        Args:
            train_matrix (scipy.sparse.csr_matrix): User-item interaction matrix.
        """
        pass
        
    @abstractmethod
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a specific user and movie.
        
        Args:
            user_id (int): Encoded user index.
            movie_id (int): Encoded movie index.
            
        Returns:
            float: Predicted rating.
        """
        pass
        
    @abstractmethod
    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return Top-N movie recommendations for a user.
        
        Args:
            user_id (int): Encoded user index.
            n (int): Number of recommendations to return.
            exclude_seen (bool): Whether to exclude movies the user has already rated.
            
        Returns:
            list: List of top N movie indices.
        """
        pass
