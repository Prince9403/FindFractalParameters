"""
Find reference value for the mse loss
If we predict ramdomly --- what the loss will be?
If we predict always (0, 0) --- what the loss will be?
"""

import numpy as np


if __name__ == "__main__":
    losses = []
    for i in range(10000):
        c0a = np.random.uniform(-1.0, 1.0)
        c1a = np.random.uniform(-1.0, 1.0)

        c0b = np.random.uniform(-1.0, 1.0)
        c1b = np.random.uniform(-1.0, 1.0)

        loss = (c0a - c0b ) ** 2 + (c1a - c1b) ** 2

        losses.append(loss)

    print(f"Mean loss (between 2 random points): {np.mean(losses):.3f}")

    losses = []
    for i in range(10000):
        c0 = np.random.uniform(-1.0, 1.0)
        c1 = np.random.uniform(-1.0, 1.0)

        loss = (c0 - 0.0) ** 2 + (c1 - 0.0) ** 2

        losses.append(loss)

    print(f"Mean loss (between random point and center): {np.mean(losses):.3f}")


