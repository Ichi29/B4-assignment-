import numpy as np

def build_matrix_A(point1s, points2):
    A= []
    for (x,y), (x_prime,y_prime) in zip(points1, points2):
        A.append([x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime, x_prime])
        A.append([0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime, y_prime])
    return np.array(A, dtype = np.float64)

def build_vector_b(points2):
    b =[]
    for (x_prime, y_prime) in (points2):
        b.append([x_prime])
        b.append([y_prime])
    return np.array(b, dtype = np.float64)

def compute_H(points1, poins2):
    A = build_matrix_A(points1, points2)
    b = build_vector_b(points2)

    h = np.linalg.inv(A.T @ A) @ A.T @ b

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)

    return H

points1 = [
    (0, 0),
    (100, 0),
    (0, 100),
    (100, 100)
]

points2 = [
    (10, 20),
    (120, 15),
    (20, 130),
    (130, 125)
]

H = compute_H(points1, points2)

print("H =")
print(H)


def transform_point(H, x, y):
    denominator = H[2,0]*x+H[2,1]*y+H[2,2]
    x_prime = (H[0,0]*x + H[0,1]*y+H[0,2])/denominator
    y_prime = (H[1,0]*x + H[2,2]*y+H[1,2])/denominator
    return x_prime, y_prime

H_inv=np.linalg.inv(H)

def get_border(H_inv, points1, points2):
    
    changed_points2 = [transform_point(H_inv, x, y) for (x, y) in points2]

    all_x = [x for (x, y) in points1] + [x for (x, y) in changed_points2]
    all_y = [y for (x, y) in points2] + [y for (x, y) in changed_points2]

    min_x= int(np.floor(min(all_x)))
    max_x= int(np.ceil(max(all_x)))
    min_y= int(np.floor(min(all_y)))
    max_x= int(np.ceil(max(all_y)))

    out_w = max_x - min_x + 1
    out=h = max_y - min_y +1

    return min_x, min_y, max_x, max_y, out_w, out_h

min_x, min_y, max_x, max_y, out_w, out_h = get_border(H_inv, points1, points2)

print(min_x, min_y, max_x, max_y)
print(out_w, out_h)