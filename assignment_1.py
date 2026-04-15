import numpy as np
import cv2
import numpy as np

def build_matrix_A(points1, points2):
    A= []
    for (x,y), (x_prime,y_prime) in zip(points1, points2):
        A.append([x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime])
        A.append([0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime])
    return np.array(A, dtype = np.float64)

def build_vector_b(points2):
    b =[]
    for (x_prime, y_prime) in (points2):
        b.append([x_prime])
        b.append([y_prime])
    return np.array(b, dtype = np.float64)

def compute_H(points1, points2):
    A = build_matrix_A(points1, points2)
    b = build_vector_b(points2)

    h = np.linalg.inv(A.T @ A) @ A.T @ b
    h = h.flatten()

    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ], dtype=np.float64)

    return H


def transform_point(H, x, y):
    denominator = H[2,0]*x+H[2,1]*y+H[2,2]
    x_prime = (H[0,0]*x + H[0,1]*y+H[0,2])/denominator
    y_prime = (H[1,0]*x + H[1,1]*y+H[1,2])/denominator
    return x_prime, y_prime

H_inv=np.linalg.inv(H)

def get_border(H_inv, points1, points2):
    
    changed_points2 = [transform_point(H_inv, x, y) for (x, y) in points2]

    all_x = [x for (x, y) in points1] + [x for (x, y) in changed_points2]
    all_y = [y for (x, y) in points1] + [y for (x, y) in changed_points2]

    min_x= int(np.floor(min(all_x)))
    max_x= int(np.ceil(max(all_x)))
    min_y= int(np.floor(min(all_y)))
    max_y= int(np.ceil(max(all_y)))

    out_w = max_x - min_x + 1
    out_h = max_y - min_y +1

    print(points1)
    print(changed_points2)

    return min_x, min_y, max_x, max_y, out_w, out_h


img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

if img1 is None:
    raise FileNotFoundError("image1.jpg が読み込めません")
if img2 is None:
    raise FileNotFoundError("image2.jpg が読み込めません")#	new file:   IMG_7002.jpeg
#	new file:   IMG_7003.jpeg
#	modified:   assignment_1.py


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

w1, h1 = 4032, 3024
w2, h2 = 4032, 3024

corners1 = [
    (0, 0),
    (w1 - 1, 0),
    (0, h1 - 1),
    (w1 - 1, h1 - 1)
]

corners2 = [
    (0, 0),
    (w2 - 1, 0),
    (0, h2 - 1),
    (w2 - 1, h2 - 1)
]

min_x, min_y, max_x, max_y, out_w, out_h = get_border(H_inv, corners1, corners2)


print("border =")
print(min_x, min_y, max_x, max_y)
print("size =")
print(out_w, out_h)