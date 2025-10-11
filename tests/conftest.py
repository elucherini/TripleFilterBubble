"""Shared fixtures and utilities for tests."""
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import tempfile
import shutil
from global_params import Params
from models import Guy, GuyId, Infobit, InfobitId, BiAdj
from utils import FastGeo, SpatialGrid


@pytest.fixture
def rng():
    """Provide a deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def test_params():
    """Provide test parameters with small values for fast tests."""
    return Params(
        numguys=10,
        numfriends=4,
        numgroups=2,
        numticks=5,
        memory=5,
        acceptance_latitude=0.3,
        acceptance_sharpness=20.0,
        max_pxcor=16.0,
        seed=42,
        posting=True,
        birth_death_probability=0.0,
        refriend_probability=0.0,
        new_info_mode="central",
        numcentral=1,
    )


@pytest.fixture
def fast_geo():
    """Provide a FastGeo instance with test parameters."""
    return FastGeo(max_pxcor=16.0, lam=0.3, k=20.0)


@pytest.fixture
def temp_data_dir():
    """Provide a temporary directory for test data that is cleaned up after the test."""
    tmpdir = tempfile.mkdtemp(prefix="tfb_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_guy(rng):
    """Provide a test Guy instance."""
    return Guy(
        id=GuyId(0),
        position=np.array([0.0, 0.0]),
        group=0,
    )


@pytest.fixture
def test_infobit(rng):
    """Provide a test Infobit instance."""
    return Infobit(
        id=InfobitId(0),
        position=np.array([1.0, 1.0]),
    )


@pytest.fixture
def biadj():
    """Provide an empty BiAdj graph."""
    return BiAdj()


@pytest.fixture
def spatial_grid():
    """Provide a SpatialGrid for testing."""
    return SpatialGrid(half_world_size=16.0, cell=0.3)


def assert_positions_close(pos1: np.ndarray, pos2: np.ndarray, rtol=1e-5, atol=1e-8):
    """Helper to assert two position arrays are close."""
    np.testing.assert_allclose(pos1, pos2, rtol=rtol, atol=atol)
