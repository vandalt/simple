import numpy as np
import pytest

import simple.utils as sut


@pytest.mark.parametrize(
    "args,shape",
    [
        ((), ()),
        ((), (5,)),
        ([1], ()),
        ((1,), ()),
        ([[1]], ()),
        ([1, 7], ()),
        ([1, 7], (2, 3, 3)),
        ([np.array([1, 7, 3])], (2, 3, 3)),
    ],
)
def test_promote_shape_compare_numpyro(args, shape):
    numpyro = pytest.importorskip("numpyro")
    promote_shapes_numpyro = numpyro.distributions.util.promote_shapes
    np.testing.assert_allclose(sut.promote_shapes(*args, shape=shape), promote_shapes_numpyro(*args, shape=shape))
