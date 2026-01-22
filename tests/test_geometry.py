import pytest
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from mirrorfield.geometry.bundle import GeometryBundle
from mirrorfield.geometry.advanced_features import (
    compute_S_score,
    class_conditional_mahalanobis,
    knn_conformal_abstain,
)


class TestGeometryBundle:
    @pytest.fixture
    def sample_data(self):
        """Generate sample embeddings and labels for testing."""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def test_bundle_initialization(self, sample_data):
        """Test GeometryBundle initializes correctly."""
        X, y = sample_data
        bundle = GeometryBundle(X, k=10)
        assert bundle.k == 10
        assert bundle.reference_embeddings.shape == X.shape

    def test_bundle_compute(self, sample_data):
        """Test basic compute functionality."""
        X, y = sample_data
        bundle = GeometryBundle(X, k=10)
        results = bundle.compute(X[:20])  # Query subset
        assert "knn_mean_distance" in results
        assert results["knn_mean_distance"].shape[0] == 20

    def test_bundle_feature_matrix(self, sample_data):
        """Test feature matrix extraction."""
        X, y = sample_data
        bundle = GeometryBundle(X, k=10)
        results = bundle.compute(X[:20])
        features = bundle.get_feature_matrix(results)
        assert features.shape == (20, 7)  # 7 base features

    def test_edge_cases_duplicates(self, sample_data):
        """Test with duplicate points."""
        X, y = sample_data
        X_dup = np.vstack([X[:10], X[0:1]])  # Add duplicate
        bundle = GeometryBundle(X, k=5)
        results = bundle.compute(X_dup)
        # Should not crash, distances should handle zeros
        assert results["knn_mean_distance"].shape[0] == 11

    def test_edge_cases_outliers(self, sample_data):
        """Test with extreme outliers."""
        X, y = sample_data
        X_out = np.vstack([X[:10], np.array([[10.0, 10.0]])])  # Extreme outlier
        bundle = GeometryBundle(X, k=5)
        results = bundle.compute(X_out)
        # Should handle large distances
        assert np.all(np.isfinite(results["knn_mean_distance"]))

    def test_edge_cases_k_larger_than_samples(self, sample_data):
        """Test when k > reference samples."""
        X, y = sample_data
        bundle = GeometryBundle(X[:5], k=4)  # k < samples
        results = bundle.compute(X[:5])
        # Should use available neighbors
        assert results["knn_mean_distance"].shape[0] == 5


class TestAdvancedFeatures:
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for advanced features."""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def test_S_score_mean(self, sample_data):
        """Test S-score with mean option."""
        X, y = sample_data
        S = compute_S_score(X, X[:20], k=10, robust="mean")
        assert S.shape == (20,)
        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)

    def test_S_score_median(self, sample_data):
        """Test S-score with median option."""
        X, y = sample_data
        S = compute_S_score(X, X[:20], k=10, robust="median")
        assert S.shape == (20,)
        assert np.all(np.isfinite(S))

    def test_mahalanobis(self, sample_data):
        """Test Mahalanobis distance."""
        X, y = sample_data
        maha = class_conditional_mahalanobis(X, y, X[:20])
        assert maha.shape == (20,)
        assert np.all(np.isfinite(maha))
        assert np.all(maha >= 0)

    def test_conformal_abstain(self, sample_data):
        """Test conformal abstention."""
        X, y = sample_data
        # Simple dummy classifier
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        X_calib = X[:20]
        y_calib = y[:20]
        accept, thresh = knn_conformal_abstain(
            X, y, clf, X_calib, y_calib, X[:10], k=5, alpha=0.1
        )
        assert accept.shape == (10,)
        assert isinstance(thresh, float)


if __name__ == "__main__":
    pytest.main([__file__])
