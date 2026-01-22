"""
Benchmark suite for geometric safety features performance.
"""

import time
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from mirrorfield.geometry import GeometryBundle


class PerformanceBenchmark:
    """Benchmark performance across different configurations."""

    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_scaling_benchmark(
        self,
        n_samples_range: List[int] = [1000, 5000, 10000],
        n_features_range: List[int] = [128, 256, 512],
        engines: List[str] = ["sklearn", "faiss"],
        k: int = 50,
    ) -> Dict:
        """Benchmark computation time vs dataset size."""
        results = []

        for n_samples in n_samples_range:
            for n_features in n_features_range:
                # Generate synthetic data
                np.random.seed(42)
                reference = np.random.randn(n_samples, n_features).astype(np.float32)
                query = np.random.randn(min(500, n_samples // 10), n_features).astype(
                    np.float32
                )

                for engine in engines:
                    try:
                        # Time initialization
                        start_init = time.perf_counter()
                        bundle = GeometryBundle(reference, k=k, engine=engine)
                        init_time = time.perf_counter() - start_init

                        # Time computation
                        start_compute = time.perf_counter()
                        features = bundle.compute(query)
                        compute_time = time.perf_counter() - start_compute

                        result = {
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "engine": engine,
                            "k": k,
                            "init_time": init_time,
                            "compute_time": compute_time,
                            "total_time": init_time + compute_time,
                            "queries_per_sec": len(query) / compute_time,
                        }
                        results.append(result)
                        print(".3f")

                    except Exception as e:
                        print(f"Failed {engine} ({n_samples}, {n_features}): {e}")

        # Save results
        self._save_results(results, "scaling_benchmark.json")
        self._generate_plot(results)
        return results

    def _save_results(self, results: List[Dict], filename: str):
        """Save benchmark results to JSON."""
        with open(self.output_dir / filename, "w") as f:
            json.dump(results, f, indent=2)

    def _generate_plot(self, results: List[Dict]):
        """Generate simple performance plot."""
        try:
            import matplotlib.pyplot as plt

            # Plot compute time vs n_samples
            fig, ax = plt.subplots(figsize=(10, 6))

            for engine in set(r["engine"] for r in results):
                engine_results = [r for r in results if r["engine"] == engine]
                sizes = [r["n_samples"] for r in engine_results]
                times = [r["compute_time"] for r in engine_results]

                ax.plot(sizes, times, "o-", label=engine)

            ax.set_xlabel("Number of Samples")
            ax.set_ylabel("Compute Time (seconds)")
            ax.set_title("Geometric Features Performance Benchmark")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig(
                self.output_dir / "performance_benchmark.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        except ImportError:
            print("Matplotlib not available, skipping plot generation")


def run_quick_benchmark():
    """Run a quick benchmark for validation."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_scaling_benchmark(
        n_samples_range=[1000, 5000],
        n_features_range=[128, 256],
        engines=["sklearn"],  # Only sklearn for quick test
        k=20,
    )
    print(f"Benchmark complete. Results saved to {benchmark.output_dir}")
    return results


if __name__ == "__main__":
    run_quick_benchmark()
