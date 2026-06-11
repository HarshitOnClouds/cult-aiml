import numpy as np

def compute_rmse(predictions, actuals):
    """
    Standard RMSE for rating prediction.
    """
    preds = np.array(predictions)
    acts = np.array(actuals)
    return np.sqrt(np.mean((preds - acts)**2))

def compute_mae(predictions, actuals):
    """
    Mean Absolute Error.
    """
    preds = np.array(predictions)
    acts = np.array(actuals)
    return np.mean(np.abs(preds - acts))

def average_precision_at_k(recommended_items, relevant_items, k=10):
    """
    Compute AP@K for a single user.
    recommended_items: ordered list of top-K movie IDs
    relevant_items: set of movie IDs the user actually rated >= 3.5
    """
    if not relevant_items:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / (i + 1)
            
    return sum_precisions / min(len(relevant_items), k)

def mean_average_precision_at_k(all_recommendations, all_relevant, k=10):
    """
    MAP@K averaged over all users.
    Only evaluates users who have at least one relevant item.
    """
    ap_scores = []
    users_excluded = 0
    
    for recs, rels in zip(all_recommendations, all_relevant):
        if not rels:
            users_excluded += 1
            continue # Exclude users with no relevant items in test set
        ap_scores.append(average_precision_at_k(recs, rels, k))
        
    print(f"MAP@{k} evaluation: excluded {users_excluded} users with 0 relevant items.")
    return np.mean(ap_scores)

def precision_at_k(recommended_items, relevant_items, k=10):
    """Precision at K"""
    if not relevant_items or not recommended_items:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / k

def recall_at_k(recommended_items, relevant_items, k=10):
    """Recall at K"""
    if not relevant_items:
        return 0.0
    hits = sum(1 for item in recommended_items[:k] if item in relevant_items)
    return hits / len(relevant_items)
