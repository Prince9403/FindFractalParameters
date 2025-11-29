import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    w0 = 1000
    h0 = 1000

    w = 2 * w0
    h = 2 * h0

    img = np.zeros(shape=(h, w))

    N = 200  # N should not be too large, as appears

    c = complex(-0.8, 0.156)
    # c = complex(-0.5125,  0.5213)
    # c = complex(-0.5251993, -0.5251993)
    # c = complex(-0.2, 0.2)
    # c = complex(-0.123, 0.745)
    # c = complex(-0.1, 0.15)

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
                img[i, j] = 0.11 + (1.0 - k / N) * 0.7
            else:
                img[i, j] = 0.1

    plt.imshow(img)
    plt.show()

