#!/bin/bash

# 課題3用スクリプト
# 水平方向の視線を -180 度から 180 度まで 10 度刻みで変化させ，
# 課題2で作成した透視投影プログラムを用いて画像を生成する．

# 使い方:
#   chmod +x generate_perspective_views.sh
#   ./generate_perspective_views.sh 入力画像 [出力ディレクトリ]
# 例:
#   ./generate_perspective_views.sh IMG_20260422_104136_012.jpg output_views

INPUT_IMAGE=$1
OUTPUT_DIR=${2:-output_views}
PYTHON_FILE="assignment_2_3d_vector_forbash.py"

FOV_X=60
FOV_Y=40
OUT_W=800
OUT_H=600

# 引数確認
if [ -z "$INPUT_IMAGE" ]; then
    echo "入力画像を指定してください"
    echo "例: ./generate_perspective_views.sh IMG_20260422_104136_012.jpg output_views"
    exit 1
fi

# ファイル存在確認
if [ ! -f "$INPUT_IMAGE" ]; then
    echo "入力画像が見つかりません: $INPUT_IMAGE"
    exit 1
fi

if [ ! -f "$PYTHON_FILE" ]; then
    echo "Pythonファイルが見つかりません: $PYTHON_FILE"
    echo "このスクリプトと assignment_2_3d_vector.py を同じディレクトリに置いてください"
    exit 1
fi

# 出力ディレクトリ作成
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# -180度から180度まで10度刻みで透視投影画像を生成
for (( angle=-180; angle<=180; angle+=10 )); do
    if [ $angle -lt 0 ]; then
        angle_name="m$((-angle))"
    else
        angle_name="p$angle"
    fi

    OUTPUT_IMAGE="$OUTPUT_DIR/perspective_${angle_name}.jpg"

    echo "angle = ${angle} deg -> $OUTPUT_IMAGE"

    python3 - "$INPUT_IMAGE" "$OUTPUT_IMAGE" "$angle" "$FOV_X" "$FOV_Y" "$OUT_W" "$OUT_H" <<'PYTHON_CODE'
import sys
import math
import cv2

from assignment_2_3d_vector import equirectangular_to_perspective_by_vector

input_path = sys.argv[1]
output_path = sys.argv[2]
angle_deg = float(sys.argv[3])
fov_x_deg = float(sys.argv[4])
fov_y_deg = float(sys.argv[5])
out_w = int(sys.argv[6])
out_h = int(sys.argv[7])

img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"画像を読み込めません: {input_path}")

# 水平方向の視線角度 theta を -pi から pi まで変化させる．
# y方向は変化させず，x-z平面上で視線方向ベクトルを作る．
theta = math.radians(angle_deg)
eye_vec = (math.sin(theta), 0.0, math.cos(theta))

out = equirectangular_to_perspective_by_vector(
    img,
    eye_vec=eye_vec,
    fov_x_deg=fov_x_deg,
    fov_y_deg=fov_y_deg,
    out_w=out_w,
    out_h=out_h
)

cv2.imwrite(output_path, out)
PYTHON_CODE

done

echo "完了しました: $OUTPUT_DIR"
