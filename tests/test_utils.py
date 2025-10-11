"""Unit tests for utility functions and classes."""
import pytest
import numpy as np
from pathlib import Path
from utils import FastGeo, SpatialGrid, FastStorage
from models import InfobitId, GuyId, Guy, Infobit
from global_params import Params
import math


class TestFastGeo:
    """Tests for FastGeo distance and probability calculations."""

    def test_dist2_zero_distance(self, fast_geo):
        """Test distance calculation when points are identical."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert fast_geo.dist2(a, b) == 0.0

    def test_dist2_simple_distance(self, fast_geo):
        """Test distance calculation for simple cases."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        # 3^2 + 4^2 = 9 + 16 = 25
        assert fast_geo.dist2(a, b) == 25.0

    def test_dist2_negative_coords(self, fast_geo):
        """Test distance calculation with negative coordinates."""
        a = np.array([-1.0, -1.0])
        b = np.array([2.0, 3.0])
        # (2-(-1))^2 + (3-(-1))^2 = 3^2 + 4^2 = 25
        assert fast_geo.dist2(a, b) == 25.0

    def test_norm_dist_basic(self, fast_geo):
        """Test normalized distance calculation."""
        a = np.array([0.0, 0.0])
        b = np.array([16.5, 0.0])
        # Distance = 16.5, norm = 1/(16.0+0.5) = 1/16.5
        # Normalized distance = 16.5 * (1/16.5) = 1.0
        result = fast_geo.norm_dist(a, b)
        assert abs(result - 1.0) < 1e-10

    def test_integration_prob_zero_distance(self, fast_geo):
        """Test integration probability when distance is zero."""
        # At d=0, probability should be 1.0
        prob = fast_geo.integration_prob_from_d2(0.0)
        assert prob == 1.0

    def test_integration_prob_at_lambda(self):
        """Test integration probability at distance = lambda."""
        # At normalized distance = lambda, probability should be 0.5
        geo = FastGeo(max_pxcor=16.0, lam=0.3, k=20.0)
        # lambda = 0.3, so unnormalized distance should be 0.3 / inv_norm
        # inv_norm = 1/16.5
        # d = 0.3 * 16.5 = 4.95
        d = 0.3 * 16.5
        d2 = d * d
        prob = geo.integration_prob_from_d2(d2)
        # At d_norm = lambda, prob = lambda^k / (lambda^k + lambda^k) = 0.5
        assert abs(prob - 0.5) < 1e-10

    def test_integration_prob_large_distance(self, fast_geo):
        """Test integration probability decreases with distance."""
        d1_squared = 1.0
        d2_squared = 100.0
        prob1 = fast_geo.integration_prob_from_d2(d1_squared)
        prob2 = fast_geo.integration_prob_from_d2(d2_squared)
        assert prob1 > prob2
        assert prob2 > 0.0  # Should never be exactly zero

    def test_integration_prob_monotonic(self, fast_geo):
        """Test that integration probability decreases monotonically with distance."""
        distances_squared = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        probs = [fast_geo.integration_prob_from_d2(d2) for d2 in distances_squared]
        # Check monotonic decrease (allowing for floating point equality at very small distances)
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], f"Probability not decreasing: {probs[i]} < {probs[i + 1]}"
            # Also check that they're not all the same
        assert probs[0] > probs[-1], "Probabilities should decrease overall"

    def test_precomputed_values(self):
        """Test that precomputed values are correct."""
        max_pxcor = 16.0
        lam = 0.3
        k = 20.0
        geo = FastGeo(max_pxcor, lam, k)

        assert geo.inv_norm == 1.0 / (max_pxcor + 0.5)
        assert geo.k == k
        assert geo.k_half == k / 2.0
        assert geo.inv_norm_pow_k == (1.0 / (max_pxcor + 0.5)) ** k
        assert geo.lam_pow_k == lam ** k


class TestSpatialGrid:
    """Tests for SpatialGrid spatial indexing."""

    def test_add_and_retrieve(self, spatial_grid):
        """Test adding and retrieving infobits from grid."""
        iid = InfobitId(0)
        pos = np.array([0.0, 0.0])
        spatial_grid.add(iid, pos)

        # Should be able to retrieve from same or neighboring cells
        neighbors = list(spatial_grid.neighbors(pos))
        assert iid in neighbors

    def test_neighbors_3x3(self, spatial_grid):
        """Test that neighbors returns infobits from 3x3 grid."""
        # Add infobits in a grid pattern
        positions = [
            (0.0, 0.0),   # center
            (0.3, 0.0),   # right neighbor cell
            (0.0, 0.3),   # top neighbor cell
            (-0.3, 0.0),  # left neighbor cell
            (0.0, -0.3),  # bottom neighbor cell
        ]
        iids = []
        for i, (x, y) in enumerate(positions):
            iid = InfobitId(i)
            iids.append(iid)
            spatial_grid.add(iid, np.array([x, y]))

        # Query center position should return all nearby infobits
        center = np.array([0.0, 0.0])
        neighbors = set(spatial_grid.neighbors(center))
        # Should find at least the center infobit
        assert InfobitId(0) in neighbors

    def test_distant_infobits_not_in_neighbors(self):
        """Test that distant infobits are not returned as neighbors."""
        grid = SpatialGrid(half_world_size=16.0, cell=0.3)

        # Add infobits far apart
        close_iid = InfobitId(0)
        far_iid = InfobitId(1)

        grid.add(close_iid, np.array([0.0, 0.0]))
        grid.add(far_iid, np.array([10.0, 10.0]))

        # Query near the close infobit
        neighbors = list(grid.neighbors(np.array([0.0, 0.0])))
        assert close_iid in neighbors
        assert far_iid not in neighbors

    def test_empty_grid(self, spatial_grid):
        """Test querying empty grid returns no neighbors."""
        neighbors = list(spatial_grid.neighbors(np.array([0.0, 0.0])))
        assert len(neighbors) == 0

    def test_negative_coordinates(self, spatial_grid):
        """Test grid works with negative coordinates."""
        iid = InfobitId(0)
        pos = np.array([-5.0, -5.0])
        spatial_grid.add(iid, pos)

        neighbors = list(spatial_grid.neighbors(pos))
        assert iid in neighbors


class TestFastStorage:
    """Tests for FastStorage compression and persistence."""

    def test_quantize(self, test_params, temp_data_dir):
        """Test position quantization."""
        test_params.run_dir = str(temp_data_dir)
        storage = FastStorage(test_params)

        # Test position at origin
        pos = np.array([0.0, 0.0])
        quantized = storage._quantize(pos)
        assert quantized.dtype == np.uint16

        # Test position at max extent
        pos_max = np.array([test_params.max_pxcor, test_params.max_pxcor])
        quantized_max = storage._quantize(pos_max)
        assert quantized_max[0] == test_params.quantization_scale
        assert quantized_max[1] == test_params.quantization_scale

    def test_quantize_disabled(self, test_params, temp_data_dir):
        """Test that quantization can be disabled."""
        test_params.quantize = False
        test_params.run_dir = str(temp_data_dir)
        storage = FastStorage(test_params)

        pos = np.array([1.5, 2.5])
        result = storage._quantize(pos)
        np.testing.assert_array_equal(result, pos)

    def test_setup_creates_directory(self, test_params, temp_data_dir):
        """Test that setup creates necessary directories and files."""
        test_params.run_dir = str(temp_data_dir / "test_run")
        storage = FastStorage(test_params)

        # Create minimal guys dict
        rng = np.random.default_rng(42)
        guys = {
            GuyId(0): Guy(GuyId(0), np.array([0.0, 0.0]), 0),
            GuyId(1): Guy(GuyId(1), np.array([1.0, 1.0]), 0),
        }

        storage.setup_writers(guys, T=5)

        # Check directory was created
        assert Path(test_params.run_dir).exists()
        assert Path(test_params.run_dir).is_dir()

        # Check meta file was created
        meta_path = Path(test_params.run_dir) / "meta.npz"
        assert meta_path.exists()

    def test_gid2idx_mapping(self, test_params, temp_data_dir):
        """Test that guy ID to index mapping is created correctly."""
        test_params.run_dir = str(temp_data_dir)
        storage = FastStorage(test_params)

        guys = {
            GuyId(5): Guy(GuyId(5), np.array([0.0, 0.0]), 0),
            GuyId(2): Guy(GuyId(2), np.array([1.0, 1.0]), 0),
            GuyId(10): Guy(GuyId(10), np.array([2.0, 2.0]), 0),
        }

        storage.setup_writers(guys, T=5)

        # Mapping should exist for all guy IDs
        assert 5 in storage.gid2idx
        assert 2 in storage.gid2idx
        assert 10 in storage.gid2idx

        # Indices should be 0-based and contiguous
        indices = sorted(storage.gid2idx.values())
        assert indices == [0, 1, 2]
