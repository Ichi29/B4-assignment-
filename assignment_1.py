import numpy as np

def transform_point(H, x, y):
    p = np.array([x, y, 1.0])
    q = H @ p
    x_prime = q[0] / q[2]
    y_prime = q[1] / q[2]
    return x_prime, y_prime

H= np.array([
    [1, 0, 100],
    [0, 1, 50],
    [0,0,1]
], dtype=float)

print(transform_point(H, 10, 20))