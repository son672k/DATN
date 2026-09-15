# Hướng dẫn train an toàn trên Kaggle

Luồng chuẩn:

```text
Habbas11 → YOLO hành vi
DDD → EfficientNet-B0 pretrain
NTHU subject-wise → CNN fine-tune → feature 1280-D → LSTM
```

Ba file cuối: `best_yolo.pt`, `best_cnn.pth`, `best_lstm.pth`.

## 0. Chuẩn bị notebook

Tạo Kaggle Dataset từ file `DMS_YOLO_CNN_LSTM_CODE_ONLY.zip`, bật GPU T4 và thêm:

1. `habbas11/dms-driver-monitoring-system`
2. `ismailnasri20/driver-drowsiness-dataset-ddd`
3. `samymesbah/nthu-dataset-ddd-multi-class`

Không upload `yolo26s.pt`; bước tìm mặt dùng MediaPipe. Copy code sang Working:

```python
from pathlib import Path
import shutil

INPUT = Path("/kaggle/input")

def find_directory(name):
    matches = [p for p in INPUT.rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]

NTHU_ROOT = find_directory("nthu-dataset-ddd-multi-class")
DDD_ROOT = find_directory("driver-drowsiness-dataset-ddd")
HABBAS_ROOT = find_directory("dms-driver-monitoring-system")
CODE_SOURCE = next(p for p in INPUT.rglob("DMS_YOLO_CNN_LSTM") if p.is_dir())
CODE_WORKING = Path("/kaggle/working/DMS_YOLO_CNN_LSTM")
if CODE_WORKING.exists():
    shutil.rmtree(CODE_WORKING)
shutil.copytree(CODE_SOURCE, CODE_WORKING)
%cd /kaggle/working/DMS_YOLO_CNN_LSTM
```

```bash
!pip install -q -r requirements.txt
!nvidia-smi
```

Tải model Face Detector chính thức dùng bởi MediaPipe Tasks:

```bash
!mkdir -p models
!wget -q -O models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
!ls -lh models/blaze_face_short_range.tflite
```

## 1. Manifest NTHU subject-wise

```python
!python scripts/01_prepare_nthu_manifest.py \
  --dataset-root "{NTHU_ROOT}" --output-dir outputs/manifests --seed 42
```

Kiểm tra `outputs/manifests/manifest_summary.json`: subject của train/val/test phải rời
nhau và mỗi split có cả `awake`, `drowsy`.

## 2. Crop mặt NTHU bằng MediaPipe

```python
!python scripts/02_crop_nthu_faces.py \
  --dataset-root "{NTHU_ROOT}" \
  --manifest outputs/manifests/nthu_manifest.csv \
  --output-root outputs/processed_nthu \
  --min-confidence 0.5 --padding 0.08
```

Xem `crop_summary.json` và `crop_failures.csv`. Tỷ lệ thành công nên ≥90%; xem trực
quan vài chục ảnh crop trước khi train. Bước train và demo đều dùng cùng detector và
padding để tránh lệch miền đầu vào.

## 3. Train YOLO hành vi

```python
!python scripts/03_train_behavior_yolo.py \
  --dataset-root "{HABBAS_ROOT}" --weights yolov8s.pt \
  --output-dir outputs/yolo --epochs 80 --batch 16 \
  --image-size 640 --patience 15
```

Giữ `outputs/yolo/best_yolo.pt`, `metrics.json`, thư mục `runs`. Nếu hết VRAM, giảm
batch xuống 8. Báo cáo Precision, Recall, mAP50 và mAP50-95.

## 4. Pretrain CNN trên DDD

```python
!python scripts/04_pretrain_cnn_ddd.py \
  --dataset-root "{DDD_ROOT}" --output-dir outputs/cnn_pretrain \
  --epochs 12 --batch 64 --learning-rate 0.0003 --patience 4
```

DDD chỉ dùng pretrain; không dùng accuracy DDD làm kết quả cuối.

## 5. Fine-tune CNN trên NTHU

```python
!python scripts/05_finetune_cnn_nthu.py \
  --manifest outputs/processed_nthu/processed_manifest.csv \
  --face-root outputs/processed_nthu/faces \
  --pretrained-checkpoint outputs/cnn_pretrain/cnn_pretrained_ddd.pth \
  --output-dir outputs/cnn_nthu --epochs 10 --batch 64 \
  --learning-rate 0.0001 --patience 4
```

Chọn model theo validation, chỉ mở test một lần để báo cáo. File cuối:
`outputs/cnn_nthu/best_cnn.pth`.

## 6. Tạo chuỗi LSTM

```python
!python scripts/06_build_lstm_sequences.py \
  --manifest outputs/processed_nthu/processed_manifest.csv \
  --face-root outputs/processed_nthu/faces \
  --cnn-checkpoint outputs/cnn_nthu/best_cnn.pth \
  --output-dir outputs/lstm_data --sequence-length 16 \
  --stride 4 --max-frame-gap 5 --batch 128
```

Kiểm tra `sequence_summary.json`: ba split có sequence, chiều feature là 1280; không
chuỗi nào trộn subject hoặc video.

## 7. Train LSTM

```python
!python scripts/07_train_lstm.py \
  --data-dir outputs/lstm_data --output-dir outputs/lstm \
  --epochs 15 --batch 64 --hidden-dim 128 --num-layers 2 \
  --learning-rate 0.0002 --patience 4
```

Giữ `outputs/lstm/best_lstm.pth`. Trong luận văn so sánh CNN riêng với CNN+LSTM trên
cùng test subject.

## 8. Đóng gói

```python
!python scripts/08_package_models.py \
  --yolo outputs/yolo/best_yolo.pt \
  --cnn outputs/cnn_nthu/best_cnn.pth \
  --lstm outputs/lstm/best_lstm.pth \
  --output-dir outputs/final_models
```

```bash
!zip -qr /kaggle/working/DMS_FINAL_MODELS.zip outputs/final_models
```

## Checklist bảo vệ

- [ ] Subject train/val/test không giao nhau.
- [ ] CNN và LSTM dùng cùng cách chia subject.
- [ ] Chuỗi LSTM không trộn subject/video và giữ đúng thứ tự frame.
- [ ] Báo cáo metric và confusion matrix cho từng model.
- [ ] Có ablation CNN so với CNN+LSTM.
- [ ] Đo FPS/latency của pipeline demo.
- [ ] Không tuyên bố MediaPipe là model do nhóm train.
- [ ] Không suy luận “không thắt dây” chỉ vì YOLO không thấy dây; cấu hình mặc định đã tắt luật này.
