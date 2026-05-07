import cv2
import numpy as np
import math
import os


def rotation_matrix_from_vectors(a, b):
    """
    ベクトル a を ベクトル b に一致させる回転行列を求める
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    # a と b がほぼ同じ向き
    if np.isclose(c, 1.0):
        return np.eye(3)

    # a と b がほぼ反対向き
    if np.isclose(c, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, axis):
            axis = np.array([0.0, 1.0, 0.0])

        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)

        K = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        return np.eye(3) + 2 * K @ K

    # ロドリゲスの回転公式
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + K + K @ K * ((1 - c) / (np.linalg.norm(v) ** 2))
    return R


def equirectangular_to_perspective_by_vector(
    img,
    eye_vec,
    fov_x_deg,
    fov_y_deg,
    out_w,
    out_h
):
    """
    正距円筒画像 img から、視線方向を3次元ベクトルで指定して透視投影画像を生成する
    """

    in_h, in_w = img.shape[:2]

    eye_vec = np.array(eye_vec, dtype=np.float64)
    if np.linalg.norm(eye_vec) == 0:
        raise ValueError("eye_vec にゼロベクトルは指定できません")

    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    dx = 2.0 * math.tan(fov_x / 2.0) / out_w
    dy = 2.0 * math.tan(fov_y / 2.0) / out_h

    base_vec = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(base_vec, eye_vec)

    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for vp in range(out_h):
        for up in range(out_w):
            x = (up - out_w / 2.0) * dx
            y = (vp - out_h / 2.0) * dy
            z = 1.0

            vec = np.array([x, y, z], dtype=np.float64)
            vec_rot = R @ vec

            xr, yr, zr = vec_rot

            theta = math.atan2(xr, zr)
            phi = -math.atan2(yr, math.sqrt(xr * xr + zr * zr))

            ue = (theta + math.pi) * in_w / (2.0 * math.pi)
            ve = (math.pi / 2.0 - phi) * in_h / math.pi

            ue_int = int(round(ue)) % in_w
            ve_int = int(round(ve))

            if 0 <= ve_int < in_h:
                out_img[vp, up] = img[ve_int, ue_int]

    return out_img


def save_image(output_dir, filename, img):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, img)
    print(f"保存しました: {path}")


if __name__ == "__main__":
    input_path = "IMG_20260422_104136_012.jpg"
    output_dir = "experiment_results_vector"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {input_path}")

    # 共通条件
    out_w = 1200
    out_h = 700
    fov_x_deg = 60
    fov_y_deg = 40

    # ============================================================
    # 実験3：回転方法の違い
    # 角度指定 theta=30, phi=20 に近い視線方向をベクトルで指定する
    # x = tan(30度), y = -tan(20度), z = 1
    # yを負にするのは，上方向を向く場合に画像座標系のyが負方向になるため
    # ============================================================
    eye_vec = (
        math.tan(math.radians(30)),
        -math.tan(math.radians(20)),
        1.0
    )

    out = equirectangular_to_perspective_by_vector(
        img,
        eye_vec=eye_vec,
        fov_x_deg=fov_x_deg,
        fov_y_deg=fov_y_deg,
        out_w=out_w,
        out_h=out_h
    )

    save_image(
        output_dir,
        "rotation_method_vector_theta30_phi20_fov60_40.jpg",
        out
    )

    print("\nベクトル指定による実験画像の生成が完了しました。")
    print(f"使用した eye_vec = {eye_vec}")
