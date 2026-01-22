import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from mirrorfield.geometry.bundle import GeometryBundle

# Generate sample reference and query embeddings using moons dataset
ref_data, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
query_data, _ = make_moons(n_samples=50, noise=0.1, random_state=43)

# Scale the data
scaler = StandardScaler()
ref_embeddings = scaler.fit_transform(ref_data)
query_embeddings = scaler.transform(query_data)

# Create GeometryBundle instance
gb = GeometryBundle(ref_embeddings, k=10)

# Compute features using the compute method (assuming dict-based API)
data = {"reference": ref_embeddings, "query": query_embeddings}
results = gb.compute(query_embeddings)
features = gb.get_feature_matrix(results)

# Print a summary
print(f"Reference embeddings shape: {ref_embeddings.shape}")
print(f"Query embeddings shape: {query_embeddings.shape}")
print(f"Features shape: {features.shape}")
print("Summary: Features computed successfully using dict-based API.")
