import numpy as np

def build_matrix_A(point1, point2):
    A= []
    for (x,y), (x_prime,y_prime) in zip(point1, point2):
        A.append([x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime, x_prime])
        A.append([0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime, y_prime])
    return np.array(A, dtype = np.float64)
