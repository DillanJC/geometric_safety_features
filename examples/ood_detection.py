"""
Example: Out-of-distribution detection using geometric features.

This script shows how to use Mahalanobis distance and S-score
for detecting OOD samples in a synthetic dataset.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from mirrorfield.geometry import GeometryBundle
from mirrorfield.geometry.advanced_features import (
    class_conditional_mahalanobis,
    compute_S_score,
)

# Generate synthetic ID and OOD data
np.random.seed(42)
n_id = 1000
n_ood = 200

# In-distribution: tight clusters
id_data = np.concatenate(
    [
        np.random.randn(n_id // 2, 10) + np.array([0, 0] + [0] * 8),
        np.random.randn(n_id // 2, 10) + np.array([3, 3] + [0] * 8),
    ]
)

# Labels for ID data (2 classes)
id_labels = np.concatenate([np.zeros(n_id // 2), np.ones(n_id // 2)])

# OOD: uniform random (far from clusters)
ood_data = np.random.uniform(-10, 10, (n_ood, 10))

print(f"ID data shape: {id_data.shape}")
print(f"OOD data shape: {ood_data.shape}")

# Initialize bundle with ID data as reference
bundle = GeometryBundle(id_data, k=50)

# Compute geometric features for combined data
combined = np.vstack([id_data, ood_data])
features = bundle.compute(combined)

# Compute advanced features
maha_scores = class_conditional_mahalanobis(id_data, id_labels, combined)
s_scores = compute_S_score(id_data, combined, k=50)

# Create labels: 0 for ID, 1 for OOD
labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

print("\nOOD Detection Performance (AUROC/AP):")
print("-" * 50)

# Evaluate each feature
feature_scores = {
    "knn_std_distance": features["knn_std_distance"],
    "mahalanobis": maha_scores,
    "S_score": s_scores,
    "knn_mean_distance": features["knn_mean_distance"],
}

for name, scores in feature_scores.items():
    # For some features, higher = more OOD-like
    if name in ["knn_std_distance", "knn_mean_distance", "mahalanobis", "S_score"]:
        # Invert if needed (higher = more ID-like)
        if name in ["knn_std_distance", "knn_mean_distance"]:
            scores = -scores

    try:
        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        print("20")
    except Exception as e:
        print("20")

print("\nTop performers:")
print("- knn_std_distance: Captures local spread (boundary uncertainty)")
print("- Mahalanobis: Class-conditional distance (distribution shift)")
print("- S_score: Density-scaled dispersion (combined signal)")


# Example usage in production
def detect_ood(query_embeddings, reference_embeddings, threshold=0.8):
    """
    Simple OOD detection pipeline.

    Returns True if sample is likely OOD.
    """
    bundle = GeometryBundle(reference_embeddings, k=50)
    features = bundle.compute(query_embeddings)

    # Use knn_std_distance as primary signal
    scores = features["knn_std_distance"]

    # Normalize to [0,1] range (simple approach)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

    # Flag high-scoring samples as OOD
    is_ood = scores_norm > threshold

    return is_ood, scores_norm


# Demo
query_sample = np.random.uniform(-5, 5, (5, 10))  # Mixed ID/OOD-like
is_ood, scores = detect_ood(query_sample, id_data)
print(f"\nQuery samples OOD detection:")
for i, (ood, score) in enumerate(zip(is_ood, scores)):
    print(".3f")
