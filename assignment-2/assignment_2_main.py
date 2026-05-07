import cv2
import numpy as np
import math


def equirectangular_to_perspective_xy_rotation(
    img,
    fov_x_deg,
    fov_y_deg,
    theta_eye_deg,  # Y軸回り：左右
    phi_eye_deg,    # X軸回り：上下
    out_w,
    out_h
):
    """
    正距円筒画像から、X軸・Y軸まわりの回転を入れた透視投影画像を生成する

    theta_eye_deg > 0 : 右方向を見る
    phi_eye_deg   > 0 : 下方向を見る
    """

    in_h, in_w = img.shape[:2]

    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    theta_eye = math.radians(theta_eye_deg)
    phi_eye = math.radians(phi_eye_deg)

    dx = 2.0 * math.tan(fov_x / 2.0) / out_w
    dy = 2.0 * math.tan(fov_y / 2.0) / out_h

    # Y軸まわりの回転行列：左右方向
    R_y = np.array([
        [ math.cos(theta_eye), 0.0, math.sin(theta_eye)],
        [ 0.0,                 1.0, 0.0                ],
        [-math.sin(theta_eye), 0.0, math.cos(theta_eye)]
    ])

    # X軸まわりの回転行列：上下方向
    R_x = np.array([
        [1.0, 0.0,               0.0              ],
        [0.0, math.cos(phi_eye), -math.sin(phi_eye)],
        [0.0, math.sin(phi_eye),  math.cos(phi_eye)]
    ])

    # 先にX軸回転、その後Y軸回転
    R = R_y @ R_x

    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for vp in range(out_h):
        for up in range(out_w):
            x = (up - out_w / 2.0) * dx
            y = (vp - out_h / 2.0) * dy
            z = 1.0

            vec = np.array([x, y, z])

            # X軸・Y軸方向に回転
            vec_rot = R @ vec

            x_rot = vec_rot[0]
            y_rot = vec_rot[1]
            z_rot = vec_rot[2]

            theta = math.atan2(x_rot, z_rot)
            phi = -math.atan2(y_rot, math.sqrt(x_rot * x_rot + z_rot * z_rot))

            ue = (theta + math.pi) * in_w / (2.0 * math.pi)
            ve = (math.pi / 2.0 - phi) * in_h / math.pi

            ue_int = int(round(ue)) % in_w
            ve_int = int(round(ve))

            if 0 <= ve_int < in_h:
                out_img[vp, up] = img[ve_int, ue_int]

    return out_img


if __name__ == "__main__":
    input_path = "IMG_20260422_104136_012.jpg"
    output_path = "perspective_xy_rotation.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {input_path}")

    out = equirectangular_to_perspective_xy_rotation(
        img,
        fov_x_deg=60,
        fov_y_deg=40,
        theta_eye_deg=0,   # 左右方向：+で右
        phi_eye_deg=20,    # 上下方向：+で下、-で上
        out_w=1200,
        out_h=700
    )

    cv2.imwrite(output_path, out)
    print(f"保存しました: {output_path}")