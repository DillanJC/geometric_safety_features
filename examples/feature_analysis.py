"""
Example: Feature analysis and correlation study.

This script reproduces the correlation analysis from the paper,
showing how geometric features correlate with uncertainty signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
from mirrorfield.geometry import GeometryBundle

# Generate synthetic dataset with known uncertainty patterns
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
train_size = 800
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train a classifier to get confidence scores
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)
confidence = probs.max(axis=1)

# Compute geometric features
bundle = GeometryBundle(X_train, k=50)
features = bundle.compute(X_test)

print(f"\nComputed {len(features)} geometric features")

# Analyze correlations with confidence (uncertainty proxy)
print("\nFeature Correlations with Model Confidence:")
print("-" * 60)
print("20")
print("-" * 60)

correlations = {}
for name, values in features.items():
    # Lower confidence = higher uncertainty
    # We expect negative correlation (higher feature value = more uncertainty)
    spearman_corr, _ = spearmanr(values, confidence)
    pearson_corr, _ = pearsonr(values, confidence)

    correlations[name] = {"spearman": spearman_corr, "pearson": pearson_corr}

    print("20")

# Find top features
top_features = sorted(
    correlations.items(), key=lambda x: abs(x[1]["spearman"]), reverse=True
)

print(f"\nTop 3 features by Spearman correlation:")
for i, (name, corr) in enumerate(top_features[:3]):
    print("2d")

# Error detection: correlate with misclassifications
errors = (clf.predict(X_test) != y_test).astype(int)

print(f"\nError Detection Performance (AUROC):")
print("-" * 40)

error_detection = {}
for name, values in features.items():
    try:
        auroc = roc_auc_score(errors, values)
        error_detection[name] = auroc
        print("20")
    except Exception as e:
        print("20")

# Identify best feature for error detection
best_feature = max(error_detection.items(), key=lambda x: x[1])
print(
    f"\nBest feature for error detection: {best_feature[0]} (AUROC: {best_feature[1]:.3f})"
)

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

feature_names = list(features.keys())[:8]
for i, name in enumerate(feature_names):
    # Scatter plot: feature value vs confidence
    sc = axes[i].scatter(
        features[name], confidence, c=errors, cmap="coolwarm", alpha=0.6
    )
    axes[i].set_xlabel(f"{name}")
    axes[i].set_ylabel("Model Confidence")
    axes[i].set_title(".3f")

plt.colorbar(sc, ax=axes[-1], label="Error (1=yes)")
plt.tight_layout()
plt.savefig("feature_correlation_analysis.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved as feature_correlation_analysis.png")

# Summary
print(f"\nSummary:")
print(f"- Dataset: {X_test.shape[0]} test samples")
print(f"- Model accuracy: {(1 - errors.mean()) * 100:.1f}%")
print(f"- Best geometric feature for uncertainty: {top_features[0][0]}")
print(f"- Best geometric feature for error detection: {best_feature[0]}")

print(f"\nKey Insight: Geometric features provide complementary signals")
print(f"to model confidence, enabling better uncertainty quantification.")
