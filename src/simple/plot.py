import matplotlib.pyplot as plt


def chainplot(chains, labels: list[str] | None = None):
    ndim = chains.shape[-1]
    if labels is None:
        labels = [f"x{i}" for i in range(ndim)]
    fig, axs = plt.subplots(ndim, 1)
    for i in range(ndim):
        axs[i].plot(chains[:, :, i], "k-", alpha=0.1)
        axs[i].set_ylabel(labels[i])
    axs[-1].set_xlabel("Steps")
    return fig, axs
