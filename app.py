import os
import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle

from src.models.svd_model import SVDModel
from src.recommendation.generator import RecommendationGenerator

# Set page config
st.set_page_config(page_title="Netflix Recommender", layout="wide")
st.title("🍿 Netflix Prize Recommendation System")
st.markdown("Interactive dashboard to explore Personalized Content Discovery.")

@st.cache_resource
def load_data_and_model():
    """Loads dataset and trains the SVD model (cached for performance)"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    train_path = os.path.join(processed_dir, 'tiny_train.parquet')
    metadata_path = os.path.join(project_root, 'data', 'raw', 'movie_titles.csv')
    
    if not os.path.exists(train_path):
        return None, None, None, None
        
    train_df = pd.read_parquet(train_path)
    
    # Load movie metadata manually to handle commas
    movies = []
    with open(metadata_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            parts = line.strip().split(',', 2)
            if len(parts) == 3:
                movies.append({'movie_id': int(parts[0]), 'year': parts[1], 'title': parts[2]})
    df_movies = pd.DataFrame(movies)
    
    with open(os.path.join(processed_dir, 'tiny_movie_encoder.pkl'), 'rb') as f:
        movie_encoder = pickle.load(f)
        
    with open(os.path.join(processed_dir, 'tiny_user_encoder.pkl'), 'rb') as f:
        user_encoder = pickle.load(f)
        
    valid_movie_ids = movie_encoder.classes_
    df_movies = df_movies[df_movies['movie_id'].isin(valid_movie_ids)].copy()
    df_movies['movie_index'] = movie_encoder.transform(df_movies['movie_id'])
    df_movies.set_index('movie_index', inplace=True)
    
    # Load sparse matrix
    user_item_train = sp.load_npz(os.path.join(processed_dir, 'tiny_user_item_train.npz'))
    
    # Train Model (SVD is fast on Tiny dataset)
    model = SVDModel(n_factors=50, n_epochs=20)
    model.fit(train_df)
    
    return model, df_movies, user_item_train, user_encoder

model, df_movies, user_item_train, user_encoder = load_data_and_model()

if model is None:
    st.error("Data not found! Please run the data pipeline (`loader.py`, `sampler.py`, `preprocessor.py`) first.")
else:
    generator = RecommendationGenerator(model, df_movies, user_item_train)
    
    # --- Sidebar ---
    st.sidebar.header("User Settings")
    
    # Get a list of valid users
    valid_users = list(range(len(user_encoder.classes_)))
    sample_user_idx = st.sidebar.number_input("Enter User Index", min_value=0, max_value=len(valid_users)-1, value=0)
    
    # --- Main Content ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🎬 Top Recommendations for User {sample_user_idx}")
        
        if st.button("Generate Recommendations"):
            with st.spinner("Finding the best movies for you..."):
                recs = generator.get_top_k(sample_user_idx, k=10, exclude_seen=True)
                
                for i, rec in enumerate(recs):
                    st.markdown(f"**{i+1}. {rec['title']}** ({rec['year']})")
                    # Note: We aren't displaying scores because SVD scores are abstract 1-5 ratings
                    
        st.markdown("---")
        st.subheader("User's Watch History")
        user_ratings = user_item_train[sample_user_idx].toarray().flatten()
        seen_indices = np.where(user_ratings > 0)[0]
        
        if len(seen_indices) > 0:
            # Show top 5 highest rated movies they watched
            user_df = pd.DataFrame({
                'movie_index': seen_indices,
                'rating': user_ratings[seen_indices]
            })
            user_df = user_df.sort_values(by='rating', ascending=False).head(10)
            
            for _, row in user_df.iterrows():
                movie_name = df_movies.loc[row['movie_index'], 'title']
                st.write(f"⭐ {row['rating']} - {movie_name}")
        else:
            st.write("This user hasn't watched anything yet (Cold Start).")
            
    with col2:
        st.subheader("🔍 Find Similar Movies")
        search_term = st.text_input("Search for a movie title (e.g. 'Matrix')")
        
        if search_term:
            matches = df_movies[df_movies['title'].str.contains(search_term, case=False, na=False)]
            if not matches.empty:
                selected_movie_id = st.selectbox("Select a movie", matches.index, format_func=lambda x: f"{matches.loc[x, 'title']} ({matches.loc[x, 'year']})")
                
                st.markdown(f"**Similar movies to {matches.loc[selected_movie_id, 'title']}:**")
                try:
                    inner_id = model.model.trainset.to_inner_iid(selected_movie_id)
                    target_vec = model.model.qi[inner_id]
                    norms = np.linalg.norm(model.model.qi, axis=1)
                    # avoid division by zero
                    norms[norms == 0] = 1e-9
                    target_norm = np.linalg.norm(target_vec)
                    if target_norm == 0:
                        target_norm = 1e-9
                    sims = np.dot(model.model.qi, target_vec) / (norms * target_norm)
                    
                    # Get top 5 highest similarities, ignoring the movie itself (which is at -1)
                    top_indices = np.argsort(sims)[-6:-1][::-1]
                    
                    for i in top_indices:
                        raw_iid = model.model.trainset.to_raw_iid(i)
                        title = df_movies.loc[raw_iid, 'title']
                        year = df_movies.loc[raw_iid, 'year']
                        st.write(f"- **{title}** ({year})")
                        
                except ValueError:
                    st.write("Not enough data to find similar movies for this specific title in our subset.")
            else:
                st.warning("No movies found.")
