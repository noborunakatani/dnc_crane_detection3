import cv2
import easyocr
import sys
import csv
import os

# ---------------------------
# 入力引数
# ---------------------------
if len(sys.argv) != 3:
    print("使い方: python read_magnification_video_complete_fixed.py <動画ファイル> <出力フォルダ>")
    sys.exit(1)

video_path = sys.argv[1]
output_dir = "C:/crane/ocr_output"

# 出力フォルダ作成
os.makedirs(output_dir, exist_ok=True)

# EasyOCR Reader 初期化
reader = easyocr.Reader(['en'])

# 動画読み込み
cap = cv2.VideoCapture(video_path)

# CSV準備
csv_path = os.path.join(output_dir, 'ocr_result_complete_fixed.csv')
csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame_number', 'ocr_value', 'confidence', 'image_file', '補完フラグ'])

# ---------------------------
# 設定
# ---------------------------
save_interval = 5  # 5フレームごとに処理
ocr_values = []    # OCR値保存
confidences = []   # 信頼度保存
補完フラグ = []    # 補完したかどうか

frame_count = 0

# ---------------------------
# OCR前処理
# ---------------------------
def preprocess_for_ocr(crop_img):
    """グレースケールのみ"""
    return cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

# ---------------------------
# 動画フレーム処理
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % save_interval != 0:
        continue

    # クロップ範囲（広めに設定）
    crop_1 = frame[973:1017, 1542:1574]   # 1桁用
    crop_2 = frame[973:1017, 1505:1574]   # 2桁用

    processed_crop_1 = preprocess_for_ocr(crop_1)
    processed_crop_2 = preprocess_for_ocr(crop_2)

    # OCR実行
    result_2 = reader.readtext(processed_crop_2, allowlist='0123456789')
    result_1 = reader.readtext(processed_crop_1, allowlist='0123456789')

    value = ''
    confidence = ''
    bbox_to_draw = None

    # 2桁優先 → 1桁
    if result_2 and result_2[0][1].isdigit() and 1 <= int(result_2[0][1]) <= 30:
        value = int(result_2[0][1])
        confidence = result_2[0][2]
        bbox_to_draw = result_2[0][0]
    elif result_1 and result_1[0][1].isdigit() and 1 <= int(result_1[0][1]) <= 30:
        value = int(result_1[0][1])
        confidence = result_1[0][2]
        bbox_to_draw = result_1[0][0]

    ocr_values.append(value)
    confidences.append(confidence if value != '' else '')
    補完フラグ.append('')  # 未補完

    # 画像に四角とOCR値表示
    if bbox_to_draw:
        (top_left, top_right, bottom_right, bottom_left) = bbox_to_draw
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 2)
        cv2.putText(frame, f'{value} ({confidence:.2f})', 
                    (top_left[0], top_left[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 画像保存
    img_filename = f"frame_{frame_count}.jpg"
    img_path = os.path.join(output_dir, img_filename)
    cv2.imwrite(img_path, frame)

cap.release()

# ---------------------------
# 未検出フレーム補完（修正版）
# ---------------------------
for i in range(len(ocr_values)):
    if ocr_values[i] == '':
        prev = ocr_values[i-1] if i > 0 else None
        next_val = ocr_values[i+1] if i < len(ocr_values)-1 else None

        # prev, next_val を整数に変換できるか確認
        try:
            prev_int = int(prev) if prev != '' else None
        except:
            prev_int = None
        try:
            next_int = int(next_val) if next_val != '' else None
        except:
            next_int = None

        # 補完ルール
        if prev_int is not None and next_int is not None:
            if prev_int < next_int:  # 上昇トレンド
                ocr_values[i] = prev_int + 1
            elif prev_int > next_int:  # 下降トレンド
                ocr_values[i] = prev_int - 1
            else:
                ocr_values[i] = prev_int  # 同じ場合
        elif prev_int is not None:
            ocr_values[i] = prev_int
        elif next_int is not None:
            ocr_values[i] = next_int
        else:
            ocr_values[i] = ''
        補完フラグ[i] = '補完済み'

# ---------------------------
# CSV書き込み
# ---------------------------
csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame_number', 'ocr_value', 'confidence', 'image_file', '補完フラグ'])

for i, val in enumerate(ocr_values):
    frame_num = (i+1) * save_interval
    img_filename = f"frame_{frame_num}.jpg"
    csv_writer.writerow([frame_num, val, confidences[i], img_filename, 補完フラグ[i]])

csv_file.close()
print("OCR処理完了（未検出補完済み）")
print(f"CSV出力: {csv_path}")
