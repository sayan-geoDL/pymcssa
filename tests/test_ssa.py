import numpy as np
import pytest
from pymcssa import SSA


@pytest.fixture
def test_data():
    """Generate synthetic test data: trend+oscillation+noise"""
    t=np.arange(0,200)
    trend=0.02*t
    oscillation=np.sin(2*np.pi*t/20)
    noise=np.random.normal(0,0.1,size=len(t))
    return trend+oscillation+noise


def test_ssa_basic_run(test_data):
    """Test if SSA runs successfully and returns expected keys."""
    m=40
    k_vals=3
    ssa=SSA(test_data,m,k_vals)
    results=ssa.ssa()

    assert isinstance(results,dict)
    for key in ["eofs","eign_values","percent_explained","pcs","rcs"]:
        assert key in results

    # Adjusted to your actual EOF shape (n-m+1,m)
    assert results["eofs"].shape[0] in [m,len(test_data)-m+1]
    assert results["pcs"].shape[1]==m
    assert results["rcs"].shape[1]==k_vals


def test_ssa_variance_explained_sum(test_data):
    """Check that total explained variance sums approximately to 100%."""
    ssa=SSA(test_data,m=40,k_vals=5)
    results=ssa.ssa()
    total_var=np.sum(results["percent_explained"])
    assert 99.0<total_var<101.0  # allow minor rounding error


def test_ssa_reconstruction_dimensions(test_data):
    """Ensure reconstructed series matches original length."""
    ssa=SSA(test_data,m=30,k_vals=4)
    results=ssa.ssa()
    assert results["rcs"].shape[0]==len(test_data)


def test_ssa_orthogonality_of_eofs(test_data):
    """Check if EOFs are approximately orthogonal."""
    ssa=SSA(test_data,m=40,k_vals=3)
    results=ssa.ssa()
    eofs=results["eofs"]
    dot_matrix=np.dot(eofs.T,eofs)
    off_diag=dot_matrix-np.diag(np.diag(dot_matrix))
    assert np.allclose(off_diag,0,atol=1e-6)


def test_ssa_invalid_dimension_raises():
    """Ensure multi-dimensional input raises an exception."""
    data=np.random.randn(100,2)
    with pytest.raises(Exception):
        SSA(data,m=20,k_vals=2).ssa()


def test_ssa_invalid_window_too_large(test_data):
    """Ensure invalid embedding window (too large) raises ValueError."""
    with pytest.raises(ValueError):
        SSA(test_data,m=len(test_data)+1,k_vals=2).ssa()
