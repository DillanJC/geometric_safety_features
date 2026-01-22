#!/usr/bin/env python3
"""
Simple validation script for geometric safety features.

Tests the core functionality on synthetic data.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from mirrorfield.geometry import GeometryBundle


def main():
    print("🔍 Geometric Safety Features Validation")
    print("=" * 50)

    # Generate synthetic data
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into reference and query
    reference = X_scaled[:150]
    query = X_scaled[150:]

    print(
        f"Reference set: {reference.shape[0]} samples, {reference.shape[1]} dimensions"
    )
    print(f"Query set: {query.shape[0]} samples, {query.shape[1]} dimensions")
    print()

    # Initialize bundle
    print("Initializing GeometryBundle...")
    bundle = GeometryBundle(reference, k=20)
    print("✓ Bundle initialized successfully")
    print()

    # Compute features
    print("Computing geometric features...")
    try:
        features = bundle.compute(query)
        print("✓ Features computed successfully")
        print(f"  Feature types: {len(features)}")
        print(f"  Features per sample: {len(features)}")
        print()
    except Exception as e:
        print(f"✗ Failed to compute features: {e}")
        return False

    # Check feature values
    print("Validating feature values...")
    all_good = True
    for name, values in features.items():
        if not np.all(np.isfinite(values)):
            print(f"✗ {name}: Contains non-finite values")
            all_good = False
        elif np.any(values < 0) and name not in [
            "local_curvature"
        ]:  # curvature can be negative
            print(
                f"⚠ {name}: Contains negative values (unexpected for distance metrics)"
            )
        else:
            print(
                f"✓ {name}: {values.shape}, finite, range [{values.min():.3f}, {values.max():.3f}]"
            )

    if not all_good:
        print("✗ Feature validation failed")
        return False

    print("✓ All features validated")
    print()

    # Test feature matrix
    print("Testing feature matrix extraction...")
    try:
        matrix = bundle.get_feature_matrix(features)
        expected_shape = (query.shape[0], len(bundle.feature_names))
        if matrix.shape == expected_shape:
            print(f"✓ Feature matrix shape correct: {matrix.shape}")
        else:
            print(f"✗ Feature matrix shape wrong: {matrix.shape} vs {expected_shape}")
            return False
    except Exception as e:
        print(f"✗ Failed to extract feature matrix: {e}")
        return False

    print()

    # Test summary
    print("Testing summary statistics...")
    try:
        summary = bundle.summarize(features)
        print(
            f"✓ Summary generated with {len(summary['feature_statistics'])} feature stats"
        )
        print(f"  Sample count: {summary['n_samples']}")
    except Exception as e:
        print(f"✗ Failed to generate summary: {e}")
        return False

    print()
    print("🎉 Validation complete! All core functionality working.")
    print()
    print("Next steps:")
    print("- Run the evaluation harness: python experiments/evaluation_harness.py")
    print(
        "- Check advanced features: python -c 'from mirrorfield.geometry.advanced_features import *; print(\"Advanced features imported\")'"
    )
    print("- Run tests: python -m pytest tests/")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
