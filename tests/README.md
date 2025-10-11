# TripleFilterBubble Test Suite

This directory contains comprehensive tests for the TripleFilterBubble simulation model.

## Test Structure

### Unit Tests
- **[test_utils.py](test_utils.py)**: Tests for `FastGeo`, `SpatialGrid`, and `FastStorage` utility classes
  - Distance calculations and integration probability
  - Spatial indexing for efficient neighbor lookup
  - Data compression and storage

- **[test_models.py](test_models.py)**: Tests for `Guy`, `Infobit`, and `BiAdj` core models
  - Agent initialization and position updates
  - Information piece tracking
  - Bidirectional adjacency graph operations

- **[test_simulation.py](test_simulation.py)**: Tests for `Simulation` class methods
  - Network generation
  - Information integration and sharing
  - Birth/death and refriending mechanics

### Integration Tests
- **[test_integration.py](test_integration.py)**: Full simulation runs with various configurations
  - Deterministic reproducibility
  - Different information modes (central, individual, select close/distant)
  - Network stability and dynamic changes
  - Storage system validation

### End-to-End Tests
- **[test_e2e.py](test_e2e.py)**: System-level validation tests
  - Simulation invariants (memory cap, bounds checking, bidirectionality)
  - Opinion dynamics and convergence
  - Network clustering by groups
  - Storage round-trip verification

### Regression Tests
- **[test_golden_regression.py](test_golden_regression.py)**: Tests against golden reference data
  - Exact match with known-good simulation outputs
  - Detects unintended behavior changes

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Fast unit tests only
pytest -m unit

# Integration tests
pytest tests/test_integration.py

# End-to-end tests
pytest -m e2e

# Exclude slow tests
pytest -m "not slow"
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test
```bash
pytest tests/test_utils.py::TestFastGeo::test_integration_prob_at_lambda
```

## Test Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.slow`: Tests that take longer to run (>5 seconds)
- `@pytest.mark.e2e`: End-to-end integration tests

## Golden Reference Data

Golden reference data provides a baseline for regression testing. To regenerate:

```bash
python tests/generate_golden_data.py
```

This creates `tests/fixtures/golden_seed42_t10.json` with metrics from a deterministic simulation run.

## Test Coverage

To run tests with coverage reporting:

```bash
pip install pytest-cov
pytest --cov=src --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html
```

## Test Fixtures

Common fixtures are defined in [conftest.py](conftest.py):
- `rng`: Deterministic random number generator
- `test_params`: Small parameter set for fast tests
- `fast_geo`: FastGeo instance
- `temp_data_dir`: Temporary directory (auto-cleaned)
- `biadj`: Empty BiAdj graph

## Adding New Tests

1. Choose appropriate test file based on what you're testing
2. Use existing fixtures where possible
3. Add appropriate markers (`@pytest.mark.slow`, etc.)
4. Name tests descriptively: `test_<what>_<condition>_<expected_result>`
5. Add docstrings explaining what is being tested

Example:
```python
def test_integration_prob_zero_distance(self, fast_geo):
    """Test integration probability when distance is zero."""
    prob = fast_geo.integration_prob_from_d2(0.0)
    assert prob == 1.0
```

## Continuous Integration

Tests can be run in CI environments. Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: |
    uv sync
    uv run pytest -v
```

## Known Issues / Notes

- Some tests are probabilistic (e.g., testing that far infobits are rarely integrated). These allow small margins for random success.
- Storage tests create compressed files that may vary slightly by platform, so we test structure rather than byte-exact matches.
- Golden data tests will fail if code behavior changes - this is intentional to catch regressions.
