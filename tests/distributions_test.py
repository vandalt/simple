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
        (3.0, 2.0),
        ([2.0, 3.0, 1.0], [3.0, 4.0]),
        ([3.0, 2.0], [2.0, 3.0]),
        ([3.0, 3.0], [1.0, 1.0]),
        (3.0, [1.0, 1.0]),
        ([3.0, 2.0], 1.0),
    ],
)
def test_uniform_low_over_high(low, high):
    # Low >= high should raise ValueError
    with pytest.raises(ValueError):
        sdist.Uniform(low=low, high=high)


# TODO: Test shape
# TODO: Test log_prob
# TODO: Test sample


# def test_uniform_log_prob():
#     uniform = sdist.Uniform(low=2.0, high=3.0)
#
#     np.testing.assert_allclose(uniform.log_prob(2.5), -np.log(1.0), rtol=1e-7)
#     np.testing.assert_equal(uniform.log_prob(4.0), -np.inf)
#     assert isinstance(uniform.log_prob(2.5), float)
#
#     # TODO: Test shapes
