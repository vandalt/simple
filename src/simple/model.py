from collections import OrderedDict
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike


class Model:
    # TODO: Specify proper type and default
    forward_model: Callable = None
    parameters: OrderedDict = None
    log_likelihood: Callable = None

    # TODO: Init function to set parameters and fwd model

    def sample_prior(
        self,
        *args,
        size: Optional[int] = None,
        seed: Optional[Union[int, np.ndarray[int]]] = None,
        sample_model: bool = True,
        **kwargs,
    ) -> np.ndarray:
        # TODO: Handle seed for multiple parameters
        samples = {}
        for parameter in self.parameters:
            samples[parameter] = self.parameters[parameter].sample(size=size, seed=seed)
        if sample_model:
            theta = np.array([samples[parameter] for parameter in samples]).T
            samples["model"] = np.array(
                [self.forward_model(theta_i, *args, **kwargs) for theta_i in theta]
            )
        return samples

    def get_parameter_names(self):
        return list(self.parameters.keys())

    def prior_transform(self, u: ArrayLike) -> ArrayLike:
        x = np.array(u)
        for i, parameter in enumerate(self.parameters):
            x[i] = self.parameters[parameter].prior_transform(u[i])
        return x

    def log_prob(self, theta: ArrayLike, *args, **kwargs):
        theta = np.asarray(theta)
        log_prior = 0.0
        for param_dist, theta_i in zip(self.parameters.values(), theta):
            log_prior += param_dist.log_prob(theta_i)
        if np.isfinite(log_prior):
            return log_prior + self.log_likelihood(theta, *args, **kwargs)
        return log_prior
