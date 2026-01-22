"""
Example: Basic feature extraction and visualization.

This script demonstrates how to use geometric-safety-features
to compute and visualize uncertainty features on synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from mirrorfield.geometry import GeometryBundle

# Generate sample data (two moons)
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into reference (training) and query (test)
reference = X_scaled[:400]
query = X_scaled[400:]

print(f"Reference shape: {reference.shape}")
print(f"Query shape: {query.shape}")

# Initialize bundle
bundle = GeometryBundle(reference, k=50)

# Compute features
features = bundle.compute(query)

print(f"Computed {len(features)} feature types")
print(f"Each feature has {len(features['knn_std_distance'])} values")

# Get feature matrix for ML
feature_matrix = bundle.get_feature_matrix(features)
print(f"Feature matrix shape: {feature_matrix.shape}")

# Simple visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

feature_names = list(features.keys())
for i, name in enumerate(feature_names):
    if i < 8:  # Plot first 8 features
        sc = axes[i].scatter(
            query[:, 0], query[:, 1], c=features[name], cmap="viridis", alpha=0.7
        )
        axes[i].set_title(f"{name}")
        plt.colorbar(sc, ax=axes[i])

plt.tight_layout()
plt.savefig("feature_visualization.png", dpi=150, bbox_inches="tight")
print("Visualization saved as feature_visualization.png")

# Summary statistics
print("\nFeature Statistics:")
for name, values in features.items():
    print(".3f")
