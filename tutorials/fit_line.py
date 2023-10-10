# %%
from collections import OrderedDict

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

import simple.distributions as sdist
import simple.model as sm

# %%
rng = np.random.default_rng(123)

x = np.sort(10 * rng.random(100))
m_true = 1.0
b_true = 0.0
truths = {"m": m_true, "b": b_true, "sigma": None}
y_true = m_true * x + b_true
yerr = 0.1 + 0.5 * rng.random(x.size)
y = y_true + 2 * yerr * rng.normal(size=x.size)

# %%
ax = plt.gca()
ax.plot(x, y_true, label="True signal")
ax.errorbar(x, y, yerr=yerr, fmt="k.", capsize=2, label="Simulated data")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.legend()
plt.show()


# %%
def gaussian(y, mu, sigma):
    return -0.5 * np.sum(((y - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


# %%
def forward_model(theta: ArrayLike, x: NDArray) -> NDArray:
    m, b = theta[:2]
    return m * x + b


def gaussian_log_likelihood(
    theta: ArrayLike,
    y: ArrayLike,
    yerr: ArrayLike,
    model: ArrayLike,
    *args,
    **kwargs,
) -> float:
    mu = model(theta, *args, **kwargs)

    jitter = theta[-1]
    sigma = np.sqrt(yerr**2 + jitter**2)

    return gaussian(mu, y, sigma)


parameters = OrderedDict({
    # TODO: Change distributions for m and sigma
    "m": sdist.Uniform(low=-2.0, high=2.0),
    "b": sdist.Uniform(low=-40.0, high=40.0),
    "sigma": sdist.Uniform(low=0.0, high=10.0),
})
mymodel = sm.Model()
mymodel.parameters = parameters
mymodel.forward_model = forward_model
mymodel.log_likelihood = gaussian_log_likelihood

# %%
prior_samples = mymodel.sample_prior(x, size=1000)
param_names = list(parameters.keys())
param_samples = {p: prior_samples[p] for p in param_names}
converted_prior_samples = {
    f"{p}": np.expand_dims(prior_samples[p], axis=0) for p in prior_samples
}
prior_inf_data = az.from_dict(converted_prior_samples)

# %%
corner.corner(prior_inf_data, var_names="~model")
plt.show()

# %%
ax = plt.gca()
ax.plot(x, y_true, label="True signal")
# Slightly higher alpha on first sample for plotting
ax.plot(x, prior_samples["model"][0], color="C1", alpha=0.5, label="Prior samples")
ax.plot(x, prior_samples["model"][1:].T, color="C1", alpha=0.1)
ax.errorbar(x, y, yerr=yerr, fmt="k.", capsize=2, label="Simulated data")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.legend()
plt.show()
plt.show()

# %%
test_point = np.array([0.2, -5.0, 1.0])
mymodel.log_prob([0.0, 1.0], y, yerr, mymodel.forward_model, x)

# %%
ax = plt.gca()
ax.plot(x, y_true, label="True signal")
ax.plot(x, mymodel.forward_model(test_point, x), label="Test model")
ax.errorbar(x, y, yerr=yerr, fmt="k.", capsize=2, label="Simulated data")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.legend()
plt.show()

# %%
import emcee

nwalkers, ndim = 32, len(test_point)
num_steps = 10_000
num_warmup = 200
pos = test_point + 1e-4 * rng.standard_normal((nwalkers, ndim))

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, mymodel.log_prob, args=(y, yerr, mymodel.forward_model, x)
)
sampler.run_mcmc(pos, num_steps, progress=True)

# %%
samples = sampler.get_chain(discard=num_warmup)

# %%
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
# TODO: method to access names
labels = list(mymodel.parameters.keys())
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")
plt.show()

# %%
inf_data = az.from_emcee(sampler, var_names=labels)
inf_data = inf_data.sel(draw=slice(num_warmup, None))

# %%
az.summary(inf_data)

# %%
az.plot_trace(inf_data)
plt.show()

# %%
corner.corner(inf_data, var_names=labels, show_titles=True, truths=truths)
plt.show()

# %%
import dynesty

sampler = dynesty.NestedSampler(
    mymodel.log_likelihood,
    mymodel.prior_transform,
    len(mymodel.parameters),
    logl_args=(y, yerr, mymodel.forward_model, x),
)

dsampler = dynesty.DynamicNestedSampler(
    mymodel.log_likelihood,
    mymodel.prior_transform,
    len(mymodel.parameters),
    logl_args=(y, yerr, mymodel.forward_model, x),
)

# %%
sampler.run_nested()

# %%
dsampler.run_nested()

# %%
from dynesty import plotting as dyplot

dyplot.traceplot(sampler.results)
plt.show()

# %%
dyplot.runplot(sampler.results, logplot=True)
plt.show()

# %%
dyplot.cornerplot(sampler.results)
plt.show()

# %%
equal_samples = sampler.results.samples_equal()
d_equal_samples = dsampler.results.samples_equal()

# %%
corner_kwargs = dict(
    hist_kwargs={"density": True},
    plot_datapoints=False,
)
fig = corner.corner(inf_data, var_names=labels, show_titles=True, truths=truths, color="k",**corner_kwargs)
corner.corner(equal_samples, labels=labels, color="b", fig=fig, **corner_kwargs)
corner.corner(d_equal_samples, labels=labels, color="r", fig=fig, **corner_kwargs)
plt.show()


# TODO: Example with ultranest
# TODO: Example with multinest
# TODO: Example with zeus
# TODO: Example with blackjax?
