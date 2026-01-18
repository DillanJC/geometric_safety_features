# Onboarding: A Guide to This Research

Welcome. This document is for researchers, reviewers, or anyone trying to understand what this repository is about and whether it's worth their time.

## The One-Sentence Summary

**Geometric features computed from k-NN neighborhoods improve AI boundary distance prediction by 4.8× more on uncertain (borderline) cases than on confident (safe) cases.**

---

## The Origin

This project started with a question: *Can we detect when an AI model is about to fail before it happens?*

The initial hypothesis—called "Dark Rivers"—proposed that embedding spaces contain discrete unstable regions identifiable by geometric signatures (low curvature + high density variation). The idea was that these regions correspond to decision boundaries where models are prone to errors.

## The Journey

I built a pipeline (`mirrorfield/geometry/`) to test this. The core approach:

1. Compute 7 k-NN geometric features for each embedding (neighborhood distances, local curvature, density variation)
2. Stratify analysis by **boundary distance**—how far a prediction is from the model's decision boundary
3. Compare: Do geometric features help more in uncertain regions than confident ones?

The key experiment (`experiments/boundary_sliced_evaluation.py`) splits predictions into three zones:
- **Safe**: High confidence, correct predictions
- **Borderline**: Near the decision boundary (uncertain)
- **Unsafe**: High confidence, wrong predictions

## The Pivot

**The Dark Rivers hypothesis was wrong.**

Testing revealed 0/220 samples met the discrete threshold criteria. Modern normalized embeddings (OpenAI, Cohere, etc.) produce smooth, uniform geometry—no discrete ridges or valleys exist.

**But something else emerged.**

While discrete detection failed, *continuous* geometric features showed a striking pattern:
- Borderline zone improvement: **+3.8%** R²
- Safe zone improvement: **+0.8%** R²
- Ratio: **4.8×**

Geometry helps most precisely where baseline methods struggle most. This isn't uniform noise—it's targeted signal.

The top predictor? `knn_std_distance` (neighborhood variance). Near decision boundaries, neighborhoods become geometrically irregular. This signature predicts instability.

## What You'll Find Here

| Location | What It Contains |
|----------|------------------|
| `docs/TECHNICAL_REPORT.md` | Full paper (~13,500 words) with methodology, results, statistics |
| `experiments/boundary_sliced_evaluation.py` | Main experiment reproducing the 4.8× finding |
| `experiments/analyze_feature_importance.py` | Which features matter and why |
| `mirrorfield/geometry/bundle.py` | Production-ready API for computing features |
| `plots/` | Publication figures (PNG + PDF) |

## How to Evaluate This Work

If you have limited time, focus on:

1. **The main claim**: `docs/TECHNICAL_REPORT.md` Section 4 (Results)
2. **Reproducibility**: Run `python experiments/boundary_sliced_evaluation.py`
3. **The falsification**: Section 5.1 documents what didn't work and why

## What I'm Looking For

I'm sharing this to get feedback on:

1. **Methodological soundness** — Is the boundary-stratified evaluation approach valid?
2. **Related work** — What have I missed in the literature on geometric uncertainty quantification?
3. **Generalization** — This was tested on one embedder (OpenAI) and one task (sentiment). What would a convincing cross-domain test look like?
4. **Next steps** — What would make this publishable vs. a curiosity?

## Quick Start

```bash
# Install
pip install -e .

# Run main experiment
python experiments/boundary_sliced_evaluation.py

# Run all tests
python experiments/test_phase_e_bundle.py
```

---

## The Honest Summary

I set out to find discrete "danger zones" in embedding space. That hypothesis failed. What I found instead was subtler: continuous geometric signatures that correlate with boundary proximity, providing targeted value in uncertain regions.

Whether this is useful depends on your problem. If your system needs to flag uncertain predictions for human review, geometry gives you 4.8× more leverage on the cases that matter most.

Questions or feedback: DillanJC91@Gmail.com or open an issue on GitHub
