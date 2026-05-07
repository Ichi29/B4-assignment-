#!/bin/bash

# 課題3用スクリプト
# 水平方向の視線を 0 → 180 → -170 → -10 度の順で変化させ，
# 並べたときにつながる透視投影画像を生成する．

# 使い方:
#   bash generate_perspective_views.sh 入力画像 [出力ディレクトリ]
# 例:
#   bash generate_perspective_views.sh IMG_20260422_104136_012.jpg output_views

INPUT_IMAGE=$1
OUTPUT_DIR=${2:-output_views}
PYTHON_FILE="assignment_2_3d_vector_forbash.py"

FOV_X=60
FOV_Y=40
OUT_W=800
OUT_H=600

if [ -z "$INPUT_IMAGE" ]; then
    echo "入力画像を指定してください"
    echo "例: bash generate_perspective_views.sh IMG_20260422_104136_012.jpg output_views"
    exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "入力画像が見つかりません: $INPUT_IMAGE"
    exit 1
fi

if [ ! -f "$PYTHON_FILE" ]; then
    echo "Pythonファイルが見つかりません: $PYTHON_FILE"
    echo "このスクリプトと $PYTHON_FILE を同じディレクトリに置いてください"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

generate_image () {
    angle=$1
    index=$2

    if [ $angle -lt 0 ]; then
        angle_label="m$((-angle))"
    else
        angle_label="p$angle"
    fi

    OUTPUT_IMAGE=$(printf "%s/perspective_%03d_%s.jpg" "$OUTPUT_DIR" "$index" "$angle_label")

    echo "angle = ${angle} deg -> $OUTPUT_IMAGE"

    python3 - "$INPUT_IMAGE" "$OUTPUT_IMAGE" "$angle" "$FOV_X" "$FOV_Y" "$OUT_W" "$OUT_H" <<'PYTHON_CODE'
import sys
import math
import cv2

from assignment_2_3d_vector_forbash import equirectangular_to_perspective_by_vector

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

theta = math.radians(angle_deg)

eye_vec = (
    math.sin(theta),
    0.0,
    math.cos(theta)
)

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
}

index=0

for (( angle=0; angle<=180; angle+=10 )); do
    generate_image "$angle" "$index"
    index=$((index + 1))
done

for (( angle=-170; angle<0; angle+=10 )); do
    generate_image "$angle" "$index"
    index=$((index + 1))
done

echo "完了しました: $OUTPUT_DIR"