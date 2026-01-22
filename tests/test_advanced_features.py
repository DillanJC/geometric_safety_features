import pytest
import numpy as np
from mirrorfield.geometry.advanced_features import (
    compute_S_score,
    class_conditional_mahalanobis,
)


class TestSScore:
    """Test S-score (density-scaled dispersion)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample embeddings and labels."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_query = np.random.randn(20, 5)
        return X_train, y_train, X_query

    def test_s_score_shape(self, sample_data):
        """S-score returns correct shape."""
        X_train, _, X_query = sample_data
        s_scores = compute_S_score(X_train, X_query, k=10)
        assert s_scores.shape == (20,)
        assert np.all(np.isfinite(s_scores))

    def test_s_score_positive(self, sample_data):
        """S-score is non-negative."""
        X_train, _, X_query = sample_data
        s_scores = compute_S_score(X_train, X_query, k=10)
        assert np.all(s_scores >= 0)

    def test_s_score_median_option(self, sample_data):
        """Median option differs from mean."""
        X_train, _, X_query = sample_data
        s_mean = compute_S_score(X_train, X_query, k=10, robust="mean")
        s_median = compute_S_score(X_train, X_query, k=10, robust="median")
        assert not np.array_equal(s_mean, s_median)


class TestMahalanobis:
    """Test class-conditional Mahalanobis distance."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with classes."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        X_query = np.random.randn(20, 5)
        return X_train, y_train, X_query

    def test_mahalanobis_shape(self, sample_data):
        """Mahalanobis returns correct shape."""
        X_train, y_train, X_query = sample_data
        maha = class_conditional_mahalanobis(X_train, y_train, X_query)
        assert maha.shape == (20,)
        assert np.all(np.isfinite(maha))

    def test_mahalanobis_non_negative(self, sample_data):
        """Mahalanobis distances are non-negative."""
        X_train, y_train, X_query = sample_data
        maha = class_conditional_mahalanobis(X_train, y_train, X_query)
        assert np.all(maha >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
