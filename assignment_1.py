import numpy as np
import cv2
import sys

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

def bilinear_interpolate(img, x, y):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    if x0 < 0 or x1 >= img.shape[1] or y0 < 0 or y1 >= img.shape[0]:
        return None

    dx = x - x0
    dy = y - y0

    I00 = img[y0, x0].astype(np.float64)
    I10 = img[y0, x1].astype(np.float64)
    I01 = img[y1, x0].astype(np.float64)
    I11 = img[y1, x1].astype(np.float64)

    I0 = (1 - dx) * I00 + dx * I10
    I1 = (1 - dx) * I01 + dx * I11
    I = (1 - dy) * I0 + dy * I1

    return np.clip(I, 0, 255).astype(np.uint8)

def transform_point(H, x, y):
    denominator = H[2,0]*x+H[2,1]*y+H[2,2]
    x_prime = (H[0,0]*x + H[0,1]*y+H[0,2])/denominator
    y_prime = (H[1,0]*x + H[1,1]*y+H[1,2])/denominator
    return x_prime, y_prime


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

def load_corresponding_points(filename):
    points1 = []
    points2 = []

    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if line == "" or line.startswith("#"):
                continue

            values = line.split()
            if len(values) != 4:
                raise ValueError(
                    f"データ不足"
                )

            x, y, x_prime, y_prime = map(float, values)

            points1.append((x, y))
            points2.append((x_prime, y_prime))

    if len(points1) < 4:
        raise ValueError("対応点不足")

    return points1, points2


img1_path = sys.argv[1]
img2_path = sys.argv[2]
points_file = sys.argv[3]
output_path = sys.argv[4]

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

points1, points2 = load_corresponding_points(points_file)
H = compute_H(points1, points2)
H_inv=np.linalg.inv(H)

print("H =")
print(H)

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

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

output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

for y in range(h1):
    for x in range(w1):
        out_x = x - min_x
        out_y = y - min_y
        output[out_y, out_x] = img1[y, x]

for out_y in range(out_h):
    for out_x in range(out_w):
        x = out_x + min_x
        y = out_y + min_y

        x2, y2 = transform_point(H, x, y)

        pixel = bilinear_interpolate(img2, x2, y2)
        if pixel is not None:
            output[out_y, out_x] = pixel

cv2.imwrite(output_path, output)
