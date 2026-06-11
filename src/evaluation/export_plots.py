import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'data', 'subsets', 'tiny_subset.parquet')
    images_dir = os.path.join(project_root, 'reports', 'images')
    
    os.makedirs(images_dir, exist_ok=True)
    
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure pipeline is run.")
        return
        
    df = pd.read_parquet(data_path)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Rating Distribution
    print("Generating Rating Distribution...")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='rating', palette='viridis', hue='rating', legend=False)
    plt.title('Overall Rating Distribution')
    plt.xlabel('Rating (Stars)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'rating_distribution.png'))
    plt.close()
    
    # 2. Ratings per user (Log scale)
    print("Generating User Activity Plot...")
    user_activity = df['user_id'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.hist(user_activity, bins=50, color='coral', log=True)
    plt.title('Ratings per User (Log Scale)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'user_activity.png'))
    plt.close()
    
    # 3. Ratings per movie (Log scale)
    print("Generating Movie Activity Plot...")
    movie_activity = df['movie_id'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.hist(movie_activity, bins=50, color='mediumseagreen', log=True)
    plt.title('Ratings per Movie (Log Scale)')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'movie_activity.png'))
    plt.close()

    print(f"Plots saved to {images_dir}")

if __name__ == "__main__":
    main()
