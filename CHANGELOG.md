# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-22

### Added
- **Core API**: `GeometryBundle` class for computing 7 geometric features
- **Advanced Features**: S-score, class-conditional Mahalanobis distance, conformal prediction
- **Performance**: Optional FAISS backend for scalable nearest neighbor search
- **Evaluation**: Comprehensive harness testing features on synthetic datasets
- **Documentation**: Complete API docs, examples, and tutorials
- **Testing**: Unit tests with 85%+ coverage and CI pipeline
- **Benchmarks**: Performance suite with scaling analysis

### Changed
- API simplified to dict-based feature returns
- Enhanced validation with reproducible evaluation templates

### Technical Details
- Rigorous evaluation identifies `knn_std_distance` as top uncertainty signal
- Boundary-stratified analysis validates improvements in high-uncertainty regions
- Compatible with Python 3.9+

## [0.1.0] - 2026-01-22

### Added
- Initial implementation of geometric safety features
- Basic evaluation on synthetic datasets
- Core functionality for AI safety diagnostics