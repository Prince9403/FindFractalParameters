import json

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    w0 = 1000
    h0 = 1000

    w = 2 * w0
    h = 2 * h0

    img = np.zeros(shape=(h, w))

    N = 200  # N should not be too large, as appears
    num_images = 1000

    dct_params = dict()

    for num in range(1, num_images + 1):
        c0 = np.random.uniform(-1.0, 1.0)
        c1 = np.random.uniform(-1.0, 1.0)

        c = complex(c0, c1)

        for i in range(h):
            for j in range(w):
                pt = complex((j - w0)/500.0, (i - h0)/500.0)
                over = False
                for k in range(N):
                    if abs(pt) > 2.0:
                        over = True
                        break
                    pt = pt * pt + c
                if over:
                    img[i, j] = 0.8
                else:
                    img[i, j] = 0.1

        plt.imsave(f"fractals_2_colors/img_{num}.jpg", img)

        dct_params[num] = [c0, c1]

        with open("fractals_2_colors/params.json", "w") as fd:
            json.dump(dct_params, fd, indent=4)
