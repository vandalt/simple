import numpy as np


def promote_shapes(*args, shape=()):
    # Adapted from numpyro.distributions.util

    if len(args) < 2 and not shape:
        return args
    else:
        shapes = [np.shape(arg) for arg in args]
        # Get maximum dimension
        num_dims = len(np.broadcast_shapes(shape, *shapes))
        return [
            np.reshape(arg, (1,) * (num_dims - len(s)) + s)
            if len(s) < num_dims
            else arg
            for arg, s in zip(args, shapes)
        ]
