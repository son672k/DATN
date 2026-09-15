# DMS — Driver Monitoring System

> **Đồ án tốt nghiệp** — Lê Hữu Sơn (22050058)

Hệ thống giám sát tài xế theo thời gian thực sử dụng YOLO + CNN + LSTM, phát hiện buồn ngủ, mất tập trung, điện thoại, thuốc lá và dây an toàn. Kiến trúc Edge–Cloud: AI chạy tại xe (Edge Simulator), dữ liệu đồng bộ về máy chủ FastAPI → MySQL, hiển thị trên Web Dashboard và Mobile App.

## Kiến trúc tổng thể

```
Camera cabin / Video
        ↓
  Edge Simulator  ──── AI: YOLO + CNN + LSTM + MediaPipe ────┐
        │                                                     │
        └──── FastAPI (Python) ──── MySQL ──── Web Dashboard  │
                                          └─── Mobile App    │
                                               (Flutter)     │
        ◄──────────────────────────── Cảnh báo real-time ────┘
```

## Ba mô hình AI

| Mô hình | Kiến trúc | Mục tiêu |
|---|---|---|
| **YOLO** | YOLOv8s | Phát hiện `closed_eye`, `open_eye`, `phone`, `cigarette`, `seatbelt` |
| **CNN** | EfficientNet-B0 | Phân loại vùng mặt → `awake / drowsy` |
| **LSTM** | LSTM 2-layer | Ổn định chuỗi 16 vector CNN 1280-D theo thời gian |

MediaPipe Face Landmarker dùng để tách vùng mặt trước CNN (không phải mô hình thứ tư cần train).

## Cấu trúc thư mục

```
DMS_YOLO_CNN_LSTM/
├── src/
│   ├── dms_inference/          # Pipeline AI inference
│   │   ├── detection/          # YOLO hành vi
│   │   ├── face/               # MediaPipe ROI + EAR/MAR/Head pose
│   │   ├── tracking/           # Làm mượt bounding box
│   │   ├── features/           # CNN + PERCLOS
│   │   ├── temporal/           # LSTM + hysteresis
│   │   ├── scoring/            # Tổng hợp cảnh báo
│   │   └── pipeline.py
│   └── dms_training/           # Module dùng khi train
├── backend/                    # FastAPI + MySQL
├── web_dashboard/              # Next.js + Tailwind
├── mobile_app/                 # Flutter
├── scripts/                    # 01→22: train → deploy → vận hành
├── configs/inference.json      # Ngưỡng AI có thể chỉnh
├── models/                     # face_landmarker.task (MediaPipe)
├── outputs/final_models/       # best_yolo.pt, best_cnn.pth, best_lstm.pth
└── data/test_videos/           # Video demo
```

## Cài đặt nhanh

### Yêu cầu

- Python 3.10+
- MySQL 8.0 (database `dms`, user `dms_app`)
- Node.js 18+ (cho Web Dashboard)
- Flutter 3.x (cho Mobile App, tuỳ chọn)

### 1. Cài Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình database

```cmd
set DMS_DATABASE_URL=mysql://dms_app:<password>@127.0.0.1:3306/dms
python scripts\13_check_mysql.py
```

### 3. Khởi động ứng dụng

```cmd
python scripts\11_run_app.py
```

- Web Dashboard: http://localhost:3000
- API docs: http://127.0.0.1:8000/docs

### 4. Chạy Edge Simulator (AI inference)

```cmd
python scripts\20_run_edge_simulator.py --trip-id <ID> --source data\test_videos\test_video.mp4 --simulate-gps
```

Xem hướng dẫn đầy đủ tại [RUN_APP.md](RUN_APP.md).

## Dataset

| Phần | Dataset | Mục đích |
|---|---|---|
| YOLO | [Habbas11 DMS (Kaggle)](https://www.kaggle.com/datasets/habbas11/dms-driver-monitoring-system) | Train hành vi |
| CNN pretrain | [Driver Drowsiness DDD (Kaggle)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) | Khởi tạo CNN |
| CNN fine-tune + LSTM | [NTHU DDD multi-class (Kaggle)](https://www.kaggle.com/datasets/samymesbah/nthu-dataset-ddd-multi-class) | Fine-tune và train LSTM |

## Thứ tự training

```
01 → Tạo manifest NTHU
02 → Crop mặt bằng MediaPipe
03 → Train YOLO (Habbas11)
04 → Pretrain CNN (DDD)
05 → Fine-tune CNN (NTHU)
06 → Trích feature, tạo chuỗi LSTM
07 → Train LSTM
08 → Đóng gói 3 checkpoint vào outputs/final_models/
```

Chi tiết từng bước tại [TRAINING_GUIDE.md](TRAINING_GUIDE.md).

## Tác giả

**Lê Hữu Sơn** — MSSV 22050058  
Đồ án tốt nghiệp, 2026
