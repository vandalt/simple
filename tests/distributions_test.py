import numpy as np
import pytest

from simple import distributions as sdist


def test_uniform_default():
    # Check default low/high attributes for uniform
    uniform = sdist.Uniform()
    assert uniform.low == 0.0
    assert uniform.high == 1.0


def test_uniform_bound_args():
    # Check that custom attributes are assigned
    uniform = sdist.Uniform(low=2.0, high=3.0)
    assert uniform.low == 2.0
    assert uniform.high == 3.0


@pytest.mark.parametrize(
    "low,high",
    [
        (2.0, 2.0),
        (3.0, 2.0),
        ([2.0, 3.0, 1.0], [3.0, 4.0]),
        ([3.0, 2.0], [2.0, 3.0]),
        ([3.0, 3.0], [1.0, 1.0]),
        (3.0, [1.0, 1.0]),
        ([3.0, 2.0], 1.0),
    ],
)
def test_uniform_low_geq_high(low, high):
    # Low >= high should raise ValueError
    with pytest.raises(ValueError):
        sdist.Uniform(low=low, high=high)


@pytest.mark.parametrize(
    "low,high",
    [
        (1.0, 2.0),
        ([2.0, 3.0], [3.0, 4.0]),
        ([2.0, 3.0], 4.0),
        (2.0, [3.0, 4.0]),
        (2.0, [[3.0, 4.0], [5.0, 8.0]]),
    ]
)
def test_uniform_shapes(low, high):
    # Check that batch shape is the same as numpyro
    uniform = sdist.Uniform(low=low, high=high)
    numpyro = pytest.importorskip("numpyro")
    numpyro_uniform = numpyro.distributions.Uniform(low=low, high=high)

    assert uniform.batch_shape == numpyro_uniform.batch_shape

# def test_uniform_log_
# TODO: Test log_prob
# TODO: Test sample
