import cv2
import numpy as np
import math


def equirectangular_to_perspective_no_rotation(img, fov_x_deg, fov_y_deg, out_w, out_h):
    """
    正距円筒画像 img から、回転なしの透視投影画像を生成する

    Parameters
    ----------
    img : np.ndarray
        入力画像（正距円筒画像）
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

    # 画角を degree -> rad に変換
    fov_x = math.radians(fov_x_deg)
    fov_y = math.radians(fov_y_deg)

    # 資料の式(4)(5): 画像面上の画素間隔 Δx, Δy
    dx = 2.0 * math.tan(fov_x / 2.0) / out_w
    dy = 2.0 * math.tan(fov_y / 2.0) / out_h

    # 出力画像を用意
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 出力画像の各画素に対して対応する入力画像の画素を求める
    for vp in range(out_h):
        for up in range(out_w):
            # 資料の式(6): 出力画像面上の点 (x, y, z)
            x = (up - out_w / 2.0) * dx
            y = (vp - out_h / 2.0) * dy
            z = 1.0

            # 資料の式(3): ベクトル -> 角度(theta, phi)
            theta = math.atan2(x, z)
            phi = -math.atan2(y, math.sqrt(x * x + z * z))

            # 資料の式(2): 角度(theta, phi) -> 正距円筒画像座標(ue, ve)
            ue = (theta + math.pi) * in_w / (2.0 * math.pi)
            ve = (math.pi / 2.0 - phi) * in_h / math.pi

            # 最近傍補間: 小数を整数に丸める
            ue_int = int(round(ue))
            ve_int = int(round(ve))

            # 横方向は360度つながっているので wrap する
            ue_int = ue_int % in_w

            # 縦方向は範囲外チェック
            if 0 <= ve_int < in_h:
                out_img[vp, up] = img[ve_int, ue_int]

    return out_img


if __name__ == "__main__":
    input_path = "IMG_20260422_104136_012.jpg"
    output_path = "perspective_no_rotation.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {input_path}")

    # 例: 水平60度、垂直40度、出力サイズ800x600
    out = equirectangular_to_perspective_no_rotation(
        img,
        fov_x_deg=60,
        fov_y_deg=40,
        out_w=800,
        out_h=600
    )

    cv2.imwrite(output_path, out)
    print(f"保存しました: {output_path}")