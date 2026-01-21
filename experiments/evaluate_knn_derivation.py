import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add the parent directory to the system path to allow importing from knn_uncertainty_derivation.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from knn_uncertainty_derivation import compute_knn_uncertainty_derivation
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# --- Data Loading (reused from boundary_sliced_evaluation.py) ---
def load_data():
    """Load sentiment classification data."""
    possible_paths = [
        Path(__file__).parent.parent,  # Current project dir
        Path(__file__).parent.parent / "runs" / "openai_3_large_test_20251231_024532",
        Path("runs/openai_3_large_test_20251231_024532"),
        Path(
            "C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532"
        ),  # Fallback
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Data not found. Tried:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
            + f"\n\nPlease ensure data files are in the runs/ directory."
        )

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    return embeddings, boundary_distances

# --- Evaluation Function for the New Metric ---
def evaluate_knn_derivation(X_train_full, X_query_full, y_query_full, n_trials=20, k=20):
    """
    Evaluate the new k-NN uncertainty derivation.
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING: k-NN Uncertainty Derivation (k={k})")
    print(f"{ '=' * 80}\n")

    print(f"Query set size: {len(y_query_full)} samples")
    print(f"Boundary distance range: [{y_query_full.min():.3f}, {y_query_full.max():.3f}]")
    print(f"Boundary distance mean: {y_query_full.mean():.3f} ± {y_query_full.std():.3f}")
    print()

    results = []

    for i in range(n_trials):
        seed = 42 + i
        
        # Split query set for evaluation (training a regressor on the uncertainty)
        # Using a fixed 80/20 split for consistency with boundary_sliced_evaluation
        X_query_train, X_query_test, y_train, y_test = train_test_split(
            X_query_full, y_query_full, test_size=0.2, random_state=seed
        )

        # Compute the new uncertainty metric for both train and test query sets
        # The X_train for compute_knn_uncertainty_derivation is the reference set.
        # It should not change per trial split.
        uncertainty_train, _, _ = compute_knn_uncertainty_derivation(X_train_full, X_query_train, k=k)
        uncertainty_test, _, _ = compute_knn_uncertainty_derivation(X_train_full, X_query_test, k=k)
        
        # Reshape for Ridge regression if it's a single feature
        uncertainty_train = uncertainty_train.reshape(-1, 1)
        uncertainty_test = uncertainty_test.reshape(-1, 1)

        # Evaluate performance of the new uncertainty metric as a predictor of boundary distance
        ridge_model = Ridge(alpha=1.0, random_state=seed)
        ridge_model.fit(uncertainty_train, y_train)
        pred_uncertainty = ridge_model.predict(uncertainty_test)
        
        r2 = r2_score(y_test, pred_uncertainty)
        mae = mean_absolute_error(y_test, pred_uncertainty)

        # Also compute direct correlation with boundary distances
        corr_uncertainty_boundary, _ = pearsonr(uncertainty_test.flatten(), y_test)
        
        results.append({
            'seed': seed,
            'r2_uncertainty': float(r2),
            'mae_uncertainty': float(mae),
            'correlation_with_boundary': float(corr_uncertainty_boundary)
        })

        if i < 3:
            print(f"  Trial {i+1:2d}: R²={r2:.3f}, MAE={mae:.3f}, Corr={corr_uncertainty_boundary:.3f}")

    print(f"  ... ({n_trials - 3} more trials)")
    print()

    # Aggregate statistics
    r2_mean = np.mean([r['r2_uncertainty'] for r in results])
    r2_std = np.std([r['r2_uncertainty'] for r in results])
    mae_mean = np.mean([r['mae_uncertainty'] for r in results])
    corr_mean = np.mean([r['correlation_with_boundary'] for r in results])
    corr_std = np.std([r['correlation_with_boundary'] for r in results])

    print("RESULTS (k-NN Uncertainty Derivation):")
    print(f"  Mean R²: {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"  Mean MAE: {mae_mean:.3f}")
    print(f"  Mean Correlation with Boundary Distance: {corr_mean:.3f} ± {corr_std:.3f}")
    print()

    return {
        'k_neighbors': k,
        'r2_mean': float(r2_mean),
        'r2_std': float(r2_std),
        'mae_mean': float(mae_mean),
        'correlation_mean': float(corr_mean),
        'correlation_std': float(corr_std),
        'trials': results
    }


def main():
    print("\n" + "="*80)
    print("EVALUATING k-NN UNCERTAINTY DERIVATION")
    print("="*80 + "\n")

    # Load data
    embeddings_full, boundary_distances_full = load_data()
    print(f"Loaded: N={len(embeddings_full)}, D={embeddings_full.shape[1]}")

    # Split into reference (X_train for compute_knn_uncertainty_derivation) and query (for evaluation)
    split_idx = int(len(embeddings_full) * 0.8)
    X_train_derivation = embeddings_full[:split_idx] # This is the reference set for the derivation
    X_query_eval = embeddings_full[split_idx:]      # This is the query set for the derivation and evaluation
    y_query_eval = boundary_distances_full[split_idx:] # Targets for the query set

    print(f"Derivation Reference Set: N={len(X_train_derivation)}")
    print(f"Evaluation Query Set: N={len(X_query_eval)}")
    print()

    # Evaluate the new k-NN uncertainty derivation
    derivation_results = evaluate_knn_derivation(X_train_derivation, X_query_eval, y_query_eval, n_trials=20, k=50)

    # Save results
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"knn_uncertainty_derivation_evaluation_{timestamp}.json"

    report = {
        'timestamp': timestamp,
        'methodology': {
            'n_trials': 20,
            'k_neighbors': derivation_results['k_neighbors'],
            'model': 'Ridge(alpha=1.0)',
            'metric': 'R² and Pearson Correlation'
        },
        'results': derivation_results
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
