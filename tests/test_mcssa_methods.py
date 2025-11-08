import numpy as np
import pytest
from pymcssa import MCSSA


@pytest.fixture
def synthetic_data():
    """Generate synthetic red-noise data for tests."""
    np.random.seed(42)
    n=300
    gamma=0.6
    data=np.zeros(n)
    for t in range(1,n):
        data[t]=gamma*data[t-1]+np.random.randn()
    return data


def test_mcssa_basic_output(synthetic_data):
    """Test that mcssa_basic runs and returns expected keys."""
    mc=MCSSA(synthetic_data,m=40)
    result=mc.mcssa_basic(up_perc=97.5,down_perc=2.5,ns=100)

    expected_keys=[
        "data_eigenvalues",
        "upper_confidence",
        "lower_confidence",
        "data_eigenvectors",
        "gamma",
        "alpha",
        "spreads",
    ]
    for key in expected_keys:
        assert key in result,f"Missing key {key} in mcssa_basic output"

    assert len(result["data_eigenvalues"])>0
    assert isinstance(result["spreads"],np.ndarray)
    assert result["spreads"].ndim==2


def test_mcssa_ensemble_output(synthetic_data):
    """Test that mcssa_ensemble runs and returns expected keys."""
    mc=MCSSA(synthetic_data,m=40)
    result=mc.mcssa_ensemble(up_perc=97.5,down_perc=2.5,ns=50)

    expected_keys=[
        "data_eigenvalues",
        "upper_confidence",
        "lower_confidence",
        "mean_surrogate_eigenvalues",
        "mean_surrogate_eigenvectors",
        "spreads",
        "alpha",
        "gamma",
    ]
    for key in expected_keys:
        assert key in result,f"Missing key {key} in mcssa_ensemble output"

    assert len(result["data_eigenvalues"])>0
    assert np.all(np.isfinite(result["data_eigenvalues"]))


def test_mcssa_procrustes_output(synthetic_data):
    """Test that mcssa_procrustes runs and returns expected keys."""
    mc=MCSSA(synthetic_data,m=40)
    result=mc.mcssa_procrustes(up_perc=97.5,down_perc=2.5,ns=30)

    expected_keys=[
        "data_eigenvalues",
        "upper_confidence",
        "lower_confidence",
        "surrogate_eigenvalues",
        "surrogate_eigenvectors",
        "procrustes_transformations",
        "alpha",
        "gamma",
    ]
    for key in expected_keys:
        assert key in result,f"Missing key {key} in mcssa_procrustes output"

    assert len(result["data_eigenvalues"])>0
    assert isinstance(result["upper_confidence"],np.ndarray) or isinstance(result["upper_confidence"],list)
    assert all(len(arr)>0 for arr in result["surrogate_eigenvalues"])


def test_invalid_parameters(synthetic_data):
    """Check that invalid parameters raise ValueErrors."""
    mc=MCSSA(synthetic_data,m=40)
    with pytest.raises(ValueError):
        mc.mcssa_basic(up_perc=101,down_perc=2.5,ns=10)
    with pytest.raises(ValueError):
        mc.mcssa_ensemble(up_perc=97.5,down_perc=97.5,ns=10)
    with pytest.raises(ValueError):
        mc.mcssa_procrustes(up_perc=97.5,down_perc=2.5,ns=0)
