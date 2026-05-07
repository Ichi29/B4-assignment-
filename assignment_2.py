import cv2
import numpy as np
import math
import sys


def create_perspective(img, theta_deg):

    in_h, in_w = img.shape[:2]

    out_w = 800
    out_h = 600

    fov_x = math.radians(60)
    fov_y = math.radians(40)

    dx = 2.0 * math.tan(fov_x / 2.0) / out_w
    dy = 2.0 * math.tan(fov_y / 2.0) / out_h

    theta0 = math.radians(theta_deg)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for vp in range(out_h):
        for up in range(out_w):

            x = (up - out_w / 2.0) * dx
            y = (vp - out_h / 2.0) * dy
            z = 1.0

            x_rot = math.cos(theta0) * x + math.sin(theta0) * z
            y_rot = y
            z_rot = -math.sin(theta0) * x + math.cos(theta0) * z

            theta = math.atan2(x_rot, z_rot)
            phi = -math.atan2(y_rot, math.sqrt(x_rot**2 + z_rot**2))

            ue = (theta + math.pi) * in_w / (2.0 * math.pi)
            ve = (math.pi / 2.0 - phi) * in_h / math.pi

            ue = int(round(ue)) % in_w
            ve = int(round(ve))

            if 0 <= ve < in_h:
                out[vp, up] = img[ve, ue]

    return out


input_path = sys.argv[1]
theta_deg = float(sys.argv[2])

img = cv2.imread(input_path)

out = create_perspective(img, theta_deg)

output_name = f"view_{int(theta_deg):03d}.jpg"

cv2.imwrite(output_name, out)