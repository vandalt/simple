import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rv_continuous, uniform

from simple.utils import promote_shapes


class Distribution:
    def log_prob(self, x: float | ArrayLike) -> float | np.ndarray:
        raise NotImplementedError

    def prior_transform(self, u: float | ArrayLike) -> float | np.ndarray:
        raise NotImplementedError

    def sample(
        self,
        size: int | None = None,
        seed: int | np.ndarray[int] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class ScipyDistribution(Distribution):
    def __init__(self, dist: rv_continuous | Callable, *args, **kwargs):
        if hasattr(dist, "dist") and hasattr(dist, "args"):
            if len(kwargs) > 0 or len(args) > 0:
                warnings.warn(
                    "The scipy distribution is 'frozen' (already instantiated). "
                    "Extra arguments will be ignored",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            self.dist = dist
        elif isinstance(dist, rv_continuous):
            self.dist = dist(*args, **kwargs)
        else:
            raise TypeError(f"Invalid type {type(dist)} for the scipy distribution.")

    def __repr__(self):
        args_tuple = self.dist.args
        kwargs_tuple = tuple(f"{k}={v}" for k, v in self.dist.kwds.items())
        signature_tuple = args_tuple + kwargs_tuple
        return f"ScipyDistribution({self.dist.dist.name}{signature_tuple})"

    def log_prob(self, x: float | ArrayLike, *args, **kwargs) -> np.ndarray:
        return self.dist.logpdf(x, *args, **kwargs)

    def prior_transform(self, u: float | ArrayLike, **kwargs) -> float | np.ndarray:
        return self.dist.ppf(u, **kwargs)

    def sample(
        self,
        size: int | None = None,
        seed: int | np.ndarray[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.dist.rvs(size=size, random_state=rng, **kwargs)


# NOTE: Example, don't use
# TODO: Move to a tutorial
# TODO: Make sure not building distributions every call: https://johannesbuchner.github.io/UltraNest/priors.html#id1
class UniformScipy(ScipyDistribution):
    def __init__(self, low: ArrayLike = 0.0, high: ArrayLike = 1.0):
        kwargs = dict(
            loc=low,
            scale=high - low,
        )
        # Create parent with super
        super().__init__(dist=uniform, **kwargs)


class Uniform(Distribution):
    def __init__(self, low: ArrayLike = 0.0, high: ArrayLike = 1.0):
        self.low, self.high = promote_shapes(low, high)
        self.batch_shape = np.broadcast_shapes(np.shape(low), np.shape(high))
        if np.any(self.low >= self.high):
            raise ValueError("'low' must be lower than 'high'")

    def log_prob(self, x: float | ArrayLike) -> np.ndarray:
        return np.where(
            np.logical_and(x >= self.low, x < self.high),
            -np.log(self.high - self.low),
            -np.inf,
        )

    def prior_transform(self, u: float) -> float:
        return self.low + u * (self.high - self.low)

    def sample(
        self,
        size: int | None = None,
        seed: int | np.ndarray[int] | None = None,
    ) -> ArrayLike:
        rng = np.random.default_rng(seed)
        if np.isscalar(size):
            size = (size,)
        shape = size + self.batch_shape if size else self.batch_shape
        return rng.uniform(low=self.low, high=self.high, size=shape)
