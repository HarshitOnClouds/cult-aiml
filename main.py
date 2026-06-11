import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle

from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.models.svd_model import SVDModel
from src.models.als_model import ALSModel
from src.evaluation.metrics import compute_rmse, mean_average_precision_at_k
from src.recommendation.generator import RecommendationGenerator

def run_experiment(prefix="tiny_"):
    project_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    train_path = os.path.join(processed_dir, f'{prefix}train.parquet')
    test_path = os.path.join(processed_dir, f'{prefix}test.parquet')
    user_item_path = os.path.join(processed_dir, f'{prefix}user_item_train.npz')
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Ensure pipeline is run.")
        return
        
    print(f"--- Running experiment on {prefix.strip('_')} dataset ---")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    user_item_train = sp.load_npz(user_item_path)
    
    # Load movie metadata mapping
    metadata_path = os.path.join(project_root, 'data', 'raw', 'movie_titles.csv')
    
    # Parse manually to handle titles with commas
    movies = []
    with open(metadata_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                movies.append({'movie_id': int(parts[0]), 'year': parts[1], 'title': parts[2]})
    df_movies = pd.DataFrame(movies)
    
    with open(os.path.join(processed_dir, f'{prefix}movie_encoder.pkl'), 'rb') as f:
        movie_encoder = pickle.load(f)
        
    # Map original IDs to encoded indices in metadata
    valid_movie_ids = movie_encoder.classes_
    df_movies = df_movies[df_movies['movie_id'].isin(valid_movie_ids)].copy()
    df_movies['movie_index'] = movie_encoder.transform(df_movies['movie_id'])
    df_movies.set_index('movie_index', inplace=True)
    
    print(f"Loaded train data: {train_df.shape[0]} ratings")
    print(f"Loaded test data: {test_df.shape[0]} ratings")
    
    # Initialize models
    models = {
        # 'UserCF': UserBasedCF(k=50),  # OOM on 200k users (320GB RAM needed)
        'ItemCF': ItemBasedCF(k=50),
        'SVD': SVDModel(n_factors=50, n_epochs=20),
        'ALS': ALSModel(factors=50, iterations=15)
    }
    
    results = []
    
    # Pre-process test data for MAP@10
    test_users = test_df['user_index'].unique()
    # Relevant items: rating >= 3.5
    relevant_items_by_user = test_df[test_df['rating'] >= 3.5].groupby('user_index')['movie_index'].apply(set).to_dict()
    all_relevant = [relevant_items_by_user.get(u, set()) for u in test_users]
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Fit
        if name == 'SVD':
            model.fit(train_df)
        else:
            model.fit(user_item_train)
            
        # Predict on Test Set (for RMSE)
        print("Predicting for RMSE...")
        test_preds = []
        for _, row in test_df.iterrows():
            pred = model.predict(row['user_index'], row['movie_index'])
            test_preds.append(pred)
            
        rmse = compute_rmse(test_preds, test_df['rating'])
        
        # Top K Recommendations (for MAP@10)
        # Note: This can be slow for many users, so we subsample for demonstration
        sample_users = test_users[:500] 
        print(f"Generating Top-10 Recs for {len(sample_users)} users (MAP@10 sample)...")
        
        generator = RecommendationGenerator(model, df_movies, user_item_train)
        
        sample_relevant = [relevant_items_by_user.get(u, set()) for u in sample_users]
        sample_recs = []
        
        for u in sample_users:
            top_k_dicts = generator.get_top_k(u, k=10, exclude_seen=True)
            top_k_indices = [x['movie_index'] for x in top_k_dicts]
            sample_recs.append(top_k_indices)
            
        map10 = mean_average_precision_at_k(sample_recs, sample_relevant, k=10)
        
        print(f"{name} Results -> RMSE: {rmse:.4f}, MAP@10: {map10:.4f}")
        results.append({'Model': name, 'RMSE': rmse, 'MAP@10': map10})
        
        if name == 'SVD':
            print("\nExample Recommendations for User 0 (SVD):")
            for rec in top_k_dicts:
                print(f"- {rec['title']} ({rec['year']})")

if __name__ == "__main__":
    run_experiment(prefix="tiny_")
