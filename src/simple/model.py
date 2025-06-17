from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Model:
    parameters: dict
    log_likelihood: Callable

    def __init__(self, parameters: dict, log_likelihood: Callable):
        self.parameters = parameters
        self._log_likelihood = log_likelihood
        # TODO: Use property via parameters?

    def keys(self):
        return list(self.parameters.keys())

    # TODO: Setter?
    def log_likelihood(self, parameters, *args, **kwargs):
        # TODO: Inherit docstring from _log_likelihood but complement to explain input
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        return self._log_likelihood(parameters, *args, **kwargs)

    def log_prior(self, parameters):
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = 0.0
        for pname, pval in parameters.items():
            pdist = self.parameters[pname]
            lp += pdist.log_prob(pval)
        return lp

    def prior_transform(self, u: ArrayLike) -> ArrayLike:
        is_dict = isinstance(u, dict)
        if is_dict:
            u = np.array(list(u.values()))
        x = np.array(u)
        for i, pdist in enumerate(self.parameters.values()):
            x[i] = pdist.prior_transform(u[i])
        if is_dict:
            x = dict(zip(self.keys(), x, strict=True))
        return x

    def nautilus_prior(self):
        from nautilus import Prior

        prior = Prior()
        for pname, pdist in self.parameters.items():
            prior.add_parameter(pname, pdist.dist)
        return prior

    def log_prob(self, parameters, *args, **kwargs):
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = self.log_prior(parameters)
        if np.isfinite(lp):
            return lp + self.log_likelihood(parameters, *args, **kwargs)
        return lp

    def get_prior_samples(self, n_samples: int):
        # TODO: Allow rng or seed for priors
        # TODO: Support emcee
        rng = np.random.default_rng()
        u = dict(
            zip(
                self.keys(),
                rng.uniform(size=(len(self.parameters), n_samples)),
                strict=True,
            )
        )
        return self.prior_transform(u)
