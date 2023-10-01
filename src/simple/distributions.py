from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from simple.utils import promote_shapes


class Distribution:
    # TODO: Register batch shape here?
    # TODO: Add simple init template?
    # TODO: Add repr and str?

    def log_prob(self, x: float) -> np.ndarray:
        # TODO: Handle ArrayLike argument and shape broadcasting
        # TODO: Implement a 'validate sample' like numpyro?
        # TODO: Write boilerplate for shape here
        raise NotImplementedError


    def prior_transform(self, u: float) -> float:
        # TODO: Handle varied shapes
        raise NotImplementedError

    def sample(
        self, size: int, seed: Optional[Union[int, np.ndarray[int]]] = None
    ) -> ArrayLike:
        # TODO: Allow other shapes, for ndim variables with proper broadcast
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(self, low: ArrayLike = 0.0, high: ArrayLike = 1.0):
        self.low, self.high = promote_shapes(low, high)
        self.batch_shape = np.broadcast_shapes(np.shape(low), np.shape(high))
        if np.any(self.low >= self.high):
            raise ValueError("'low' must be lower than 'high'")

    def log_prob(self, x: Union[float, ArrayLike]) -> np.ndarray:
        return np.where(
            np.logical_and(x >= self.low, x < self.high),
            -np.log(self.high - self.low),
            -np.inf
        )

    def prior_transform(self, u: float) -> float:
        # TODO: Handle varied shapes, take example on logprob
        return self.low + u * (self.high - self.low)

    def sample(
        self, size: Optional[int] = None, seed: Optional[Union[int, np.ndarray[int]]] = None
    ) -> ArrayLike:
        # TODO: Allow other shapes, for ndim variables with proper broadcast
        rng = np.random.default_rng(seed)
        if np.isscalar(size):
            size = (size,)
        shape = size + self.batch_shape if size else self.batch_shape
        return rng.uniform(low=self.low, high=self.high, size=shape)
