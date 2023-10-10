from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import uniform

from simple.utils import promote_shapes


class Distribution:
    # TODO: Add repr and str?

    def log_prob(self, x: float) -> np.ndarray:
        raise NotImplementedError

    def prior_transform(self, u: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        raise NotImplementedError

    def sample(
        self, size: int, seed: Optional[Union[int, np.ndarray[int]]] = None
    ) -> ArrayLike:
        raise NotImplementedError


# TODO: Document that this is slower, and how it works
class ScipyDistribution(Distribution):
    def __init__(self, scipy_dist, **kwargs):
        self.scipy_dist = scipy_dist(**kwargs)

    def log_prob(self, x: Union[float, ArrayLike], **kwargs) -> np.ndarray:
        return self.scipy_dist.logpdf(x, **kwargs)

    def prior_transform(self, u: float, **kwargs) -> float:
        return self.scipy_dist.ppf(u, **kwargs)

    def sample(
        self,
        size: Optional[int] = None,
        seed: Optional[Union[int, np.ndarray[int]]] = None,
        **kwargs,
    ) -> ArrayLike:
        rng = np.random.default_rng(seed)
        return self.scipy_dist.rvs(size=size, random_state=rng, **kwargs)


# NOTE: Example, don't use
# TODO: Move to a tutorial
class UniformScipy(ScipyDistribution):
    def __init__(self, low: ArrayLike = 0.0, high: ArrayLike = 1.0):
        kwargs = dict(
            loc=low,
            scale=high - low,
        )
        # Create parent with super
        super().__init__(scipy_dist=uniform, **kwargs)


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
            -np.inf,
        )

    def prior_transform(self, u: float) -> float:
        return self.low + u * (self.high - self.low)

    def sample(
        self,
        size: Optional[int] = None,
        seed: Optional[Union[int, np.ndarray[int]]] = None,
    ) -> ArrayLike:
        rng = np.random.default_rng(seed)
        if np.isscalar(size):
            size = (size,)
        shape = size + self.batch_shape if size else self.batch_shape
        return rng.uniform(low=self.low, high=self.high, size=shape)
