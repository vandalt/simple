from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from nautilus import Prior


class Model:
    parameters: dict
    log_likelihood: Callable

    def __init__(self, parameters: dict, log_likelihood: Callable):
        self.parameters = parameters
        self._log_likelihood = log_likelihood
        # Attempt to make log-likelihood inherit docstring.
        # Works at runtime but not for static (LSP) tools
        self.log_likelihood.__func__.__doc__ = self._log_likelihood.__doc__

    def _log_likelihood(self, parameters, *args, **kwargs) -> float:
        raise NotImplementedError(
            "log_likelihood must be passed to init or _log_likelihood must be "
            "implemented by subclasses."
        )

    def keys(self) -> list[str]:
        """List of parameter names (dictionary keys)"""
        return list(self.parameters.keys())

    def log_likelihood(self, parameters, *args, **kwargs):
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        return self._log_likelihood(parameters, *args, **kwargs)

    def log_prior(self, parameters: dict | ArrayLike) -> float:
        """Log of the prior probability for the model

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as Model.keys()
        :return: Log-prior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = 0.0
        for pname, pval in parameters.items():
            pdist = self.parameters[pname]
            lp += pdist.log_prob(pval)
        return lp

    def prior_transform(self, u: ArrayLike | dict) -> np.ndarray | dict:
        """Prior transform of the model

        Takes samples from a uniform distribution between 0 and 1
        for all parameters and returns samples transformed according to the prior.

        :param u: Samples from the uniform distribution. Can be a dict or an array
                  ordered as Model.keys()
        :return: Prior samples, as a dict or an array depending on the input type.
        """
        is_dict = isinstance(u, dict)
        if is_dict:
            u = np.array(list(u.values()))
        x = np.array(u)
        for i, pdist in enumerate(self.parameters.values()):
            x[i] = pdist.prior_transform(u[i])
        if is_dict:
            x = dict(zip(self.keys(), x, strict=True))
        return x

    def nautilus_prior(self) -> "Prior":
        """Builds and return a `nautilus.Prior` for the model.

        :return: Nautilus Prior object.
        """
        from nautilus import Prior

        prior = Prior()
        for pname, pdist in self.parameters.items():
            prior.add_parameter(pname, pdist.dist)
        return prior

    def log_prob(self, parameters: dict | ArrayLike, *args, **kwargs) -> float:
        """Log posterior probability for the model

        :param parameters: Parameters as a dict or an array ordered as Model.keys()
        :return: Log posterior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = self.log_prior(parameters)
        if np.isfinite(lp):
            return lp + self.log_likelihood(parameters, *args, **kwargs)
        return lp

    def get_prior_samples(self, n_samples: int, fmt: str = "dict") -> dict:
        """Generate prior samples

        :param n_samples: Number of samples
        :param fmt: Format of the samples (dict or array)
        :return: Dictionary of prior samples
        """
        rng = np.random.default_rng()
        u = rng.uniform(size=(len(self.parameters), n_samples))
        if fmt == "dict":
            u = dict(zip(self.keys(), u, strict=True))
        elif fmt != "array":
            raise ValueError(f"Invalid format: {fmt}. Use 'dict' or 'array'.")
        return self.prior_transform(u)


class ForwardModel(Model):
    """A model whose likelihood calls a forward model as the mean."""

    forward: Callable

    def __init__(self, parameters: dict, log_likelihood: Callable, forward: Callable):
        super().__init__(parameters, log_likelihood)
        self._forward = forward
        self.forward.__func__.doc__ = self._forward.__doc__

    def forward(self, parameters: dict | ArrayLike, *args, **kwargs) -> np.ndarray:
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        return self._forward(parameters, *args, **kwargs)

    def get_prior_pred(self, n_samples: int, *args, **kwargs) -> np.ndarray:
        prior_params = self.get_prior_samples(n_samples, fmt="array")
        pred = []
        for p in prior_params.T:
            pred.append(self.forward(p, *args, **kwargs))
        return np.array(pred)


class GaussianForwardModel(ForwardModel):
    def __init__(
        self,
        parameters: dict,
        forward: Callable,
        sigma_in_model: bool = False,
    ):
        """Initialize a Gaussian forward model.

        :param parameters: Dictionary of parameters with their prior distributions.
        :param forward: Callable that computes the forward model.
                        Should accept a dictionary of parameters.
        :param sigma_in_model: Boolean flag to allow usage of 'sigma' (extra error term)
                               in the forward model.
        """
        if "sigma" not in parameters:
            raise ValueError(
                "The extra error term 'sigma' must be a parameter in the model."
            )
        super().__init__(parameters, self._log_likelihood, forward)
        self.sigma_in_model = sigma_in_model
        self._sigma_forward_checked = False
        self._in_sigma_check = False

    def _test_forward_sigma(self, *args, **kwargs):
        self._in_sigma_check = True
        if self.sigma_in_model:
            return

        test_prior_point = self.get_prior_samples(1)
        del test_prior_point["sigma"]
        try:
            self.forward(test_prior_point, *args, **kwargs)
        except KeyError as e:
            if "sigma" in str(e):
                raise ValueError(
                    "Forward model uses the extra error term 'sigma' as a parameter."
                    "This is generally not desired. If it is, set sigma_in_model=True."
                ) from e
            raise e
        self._sigma_forward_checked = True
        self._in_sigma_check = False

    def forward(self, parameters: dict | ArrayLike, *args, **kwargs):
        if not self._sigma_forward_checked and not self._in_sigma_check:
            self._test_forward_sigma(*args, **kwargs)
        return super().forward(parameters, *args, **kwargs)

    def _log_likelihood(
        self, parameters: dict, data: np.ndarray, err: np.ndarray, *args, **kwargs
    ) -> float:
        """Gaussian log-likelihood for a forward model

        This method calls the forward model with `parameters`.
        The output is passed through a gaussian likelihood with `data` and `err`.
        If `parmaters` contains `sigma`, this parameter is added in quadrature to `err`.

        :param parameters: Parameter dictionary
        :param data: Data array
        :param err: Error array
        :return: Gaussian log-likelihood
        """
        mu = self.forward(parameters, *args, **kwargs)
        sigma = np.sqrt(parameters["sigma"] ** 2 + err**2)
        return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))
