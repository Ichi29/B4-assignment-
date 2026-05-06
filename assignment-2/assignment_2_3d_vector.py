import cv2
import numpy as np
import math


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
        # a と直交する適当な軸を作る
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, axis):
            axis = np.array([0.0, 1.0, 0.0])

        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)

        # 180度回転の回転行列
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

    Parameters
    ----------
    img : np.ndarray
        入力画像（正距円筒画像）
    eye_vec : tuple or list
        視線方向ベクトル。例: (0, 0, 1), (1, 0, 1), (0, -1, 1)
    fov_x_deg : float
        水平方向画角 [deg]
    fov_y_deg : float
        垂直方向画角 [deg]
    out_w : int
        出力画像の幅
    out_h : int
        出力画像の高さ

    Returns
    -------
    out_img : np.ndarray
        生成した透視投影画像
    """

    in_h, in_w = img.shape[:2]

    # 視線方向ベクトルの確認
    eye_vec = np.array(eye_vec, dtype=np.float64)
    if np.linalg.norm(eye_vec) == 0:
        raise ValueError("eye_vec にゼロベクトルは指定できません")

    # 画角を degree -> rad に変換
    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    # 画像面上の画素間隔
    dx = 2.0 * math.tan(fov_x / 2.0) / out_w
    dy = 2.0 * math.tan(fov_y / 2.0) / out_h

    # 基準視線方向 z軸正方向を、指定された視線方向に向ける回転行列
    base_vec = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(base_vec, eye_vec)

    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for vp in range(out_h):
        for up in range(out_w):
            # 回転前の画像面上の3次元ベクトル
            x = (up - out_w / 2.0) * dx
            y = (vp - out_h / 2.0) * dy
            z = 1.0

            vec = np.array([x, y, z], dtype=np.float64)

            # 視線方向に合わせて回転
            vec_rot = R @ vec

            xr, yr, zr = vec_rot

            # ベクトル -> 角度(theta, phi)
            theta = math.atan2(xr, zr)
            phi = -math.atan2(yr, math.sqrt(xr * xr + zr * zr))

            # 角度(theta, phi) -> 正距円筒画像座標(ue, ve)
            ue = (theta + math.pi) * in_w / (2.0 * math.pi)
            ve = (math.pi / 2.0 - phi) * in_h / math.pi

            ue_int = int(round(ue)) % in_w
            ve_int = int(round(ve))

            if 0 <= ve_int < in_h:
                out_img[vp, up] = img[ve_int, ue_int]

    return out_img


if __name__ == "__main__":
    input_path = "IMG_20260422_104136_012.jpg"
    output_path = "perspective_by_vector.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {input_path}")

    # 例1: 正面方向
    # eye_vec = (0, 0, 1)

    # 例2: 右方向を向く
    # eye_vec = (1, 0, 1)

    # 例3: 上方向を向く
    # y軸は画像下方向が正なので、上を向きたい場合は y を負にする
    eye_vec = (1, -0.3, 1)

    out = equirectangular_to_perspective_by_vector(
        img,
        eye_vec=eye_vec,
        fov_x_deg=60,
        fov_y_deg=40,
        out_w=800,
        out_h=600
    )

    cv2.imwrite(output_path, out)
    print(f"保存しました: {output_path}")