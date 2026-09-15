# KHỞI ĐỘNG DMS — KIẾN TRÚC CHÍNH THỨC

Kiến trúc demo duy nhất:

`Cabin camera/video → Edge Simulator → FastAPI → MySQL → Web/Mobile quản lý`

## 1. Kiểm tra MySQL

MySQL80 phải đang chạy và đã có database `dms`, user `dms_app`.

```cmd
set DMS_DATABASE_URL=mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms
python scripts\13_check_mysql.py
```

Kết quả phải có `"status": "PASS"`.

Sau lần cập nhật Edge đầu tiên, chạy một lần (backup trước):

```cmd
python scripts\15_backup_mysql.py
python scripts\17_migrate_edge_devices.py
```

Sau lần cập nhật kiểu dữ liệu MySQL, chạy một lần:

```cmd
python scripts\15_backup_mysql.py
python scripts\21_optimize_mysql_schema.py --apply
python scripts\22_normalize_relationships.py
python scripts\22_normalize_relationships.py --apply --repair-legacy
python scripts\16_smoke_mysql.py
```

Migration giữ dữ liệu hiện có, chuyển thời gian sang UTC `DATETIME(6)`, payload
sang `JSON`, cờ logic sang `TINYINT(1)`, số đo sang `DOUBLE`, đồng thời bổ sung
chỉ mục và khóa ngoại. Có thể chạy lại lệnh mà không tạo thay đổi trùng lặp.
Script 22 kiểm tra quan hệ trước, sau đó loại bỏ `driver_id`, `vehicle_id` dư thừa
khỏi `monitoring_sessions` và `vehicle_id` khỏi `vehicle_locations`. Nếu phát hiện
phiên cũ chưa có chuyến nhưng còn xác định được tài xế, tùy chọn `--repair-legacy`
tạo chuyến lưu trữ trước khi chuẩn hóa. Dữ liệu sai tài xế hoặc sai xe vẫn làm script dừng.

## 2. Khởi động backend và web

Trong cùng cửa sổ CMD:

```cmd
set DMS_DATABASE_URL=mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms
python scripts\11_run_app.py
```

- Web Dashboard: http://localhost:3000
- API docs: http://127.0.0.1:8000/docs

## 3. Luồng demo đúng

1. Supervisor đăng nhập.
2. Tạo tài xế, xe, tuyến và chuyến; chọn tài xế cùng xe ngay trong cửa sổ tạo chuyến.
3. Chuyển chuyến sang `running`.
4. Chọn cabin camera hoặc video trong Edge Simulator.
5. AI xử lý YOLO + CNN + LSTM và EAR/MAR/Head Pose.
6. Máy chạy Edge Simulator đọc cảnh báo bằng giọng nói; nếu Windows TTS không khả dụng thì phát beep.
7. MySQL lưu phiên, metric, cảnh báo và GPS.
8. Web/Mobile quản lý xem và xử lý cảnh báo; tài xế chỉ xem dữ liệu của mình.

Mỗi cảnh báo mới tạo snapshot và clip MP4 dài tối đa 8 giây trong
`outputs/backend_sessions/<session_id>/`. Clip gồm khoảng 3 giây trước cảnh báo
và 5 giây sau cảnh báo; phiên kết thúc sớm thì clip ngắn hơn.

Mobile App không dùng camera và không tạo phiên AI.

Edge Simulator xác thực bằng hai header `X-Edge-Device-ID` và
`X-Edge-API-Key`. Supervisor tạo thiết bị tại `/docs` qua
`POST /api/admin/edge-devices`; API key chỉ được trả về tại lần tạo này.

Mở CMD thứ hai và đặt thông tin xác thực Edge (không ghi khóa vào source code):

```cmd
set DMS_API_BASE=http://127.0.0.1:8000
set DMS_EDGE_DEVICE_ID=edge-car-001
set DMS_EDGE_API_KEY=<api-key-vừa-cấp>
```

### Cách nhanh khi demo

Nhấp đúp [START_EDGE_DEMO.vbs](START_EDGE_DEMO.vbs), sau đó:

1. Nhập mã thiết bị và API key.
2. Bấm **Tải chuyến**; Launcher chỉ hiện chuyến đang chạy thuộc đúng xe của Edge.
3. Chọn video hoặc đánh dấu webcam `camera:0`.
4. Bấm **Bắt đầu Edge**. Dùng nút **Dừng** để kết thúc an toàn.

Launcher không ghi API key vào tệp cấu hình. Có thể đặt `DMS_EDGE_API_KEY` trong
cửa sổ lệnh trước khi mở Launcher hoặc nhập lại khóa khi chạy. Chỉ đánh dấu
**Kết thúc phiên cũ của thiết bị** khi chắc chắn phiên trước là phiên mồ côi;
Launcher sẽ hỏi lại trước khi thay thế.

Các lệnh dưới đây là cách chạy thủ công khi cần kiểm tra kỹ thuật.

Chạy AI tại Edge từ video và đồng thời phát GPS mô phỏng:

```cmd
python scripts\20_run_edge_simulator.py --trip-id <ID_CHUYEN> --source data\test_videos\test_video.mp4 --simulate-gps
```

Hoặc dùng webcam cabin:

```cmd
python scripts\20_run_edge_simulator.py --trip-id <ID_CHUYEN> --source camera:0 --simulate-gps
```

Thêm `--device cuda` nếu máy Edge có CUDA; thêm `--save-video` khi cần lưu toàn
bộ video đã chú thích. Dừng bằng `Ctrl+C`; phiên được ghi trạng thái `stopped`.
Edge gửi heartbeat, metric, khung xem trước, cảnh báo, snapshot và clip bằng chứng;
FastAPI xác thực, lưu MySQL và chuyển dữ liệu cho Web/Mobile.
Tệp tạm/phần tổng kết tại Edge nằm trong `outputs/edge_sessions/<session_id>`;
có thể đổi bằng biến `DMS_EDGE_OUTPUT_ROOT`.

Dashboard không khởi chạy camera hoặc mô hình. Endpoint cũ `/api/sessions/start`
bị khóa mặc định để tránh chạy nhầm AI trong FastAPI. Chỉ bật
`DMS_ALLOW_BACKEND_INFERENCE=true` khi cần chẩn đoán tương thích mã cũ.

Nếu chỉ cần kiểm tra kết nối thiết bị mà không chạy AI, dùng:

```cmd
python scripts\18_run_edge_client.py --once
```

Nếu khóa bị lộ, gọi
`POST /api/admin/edge-devices/{device_id}/rotate-key` và cập nhật lại biến môi trường.

Để demo GPS theo tuyến cố định (dữ liệu được ghi rõ là `gps-simulated`), chuyến
phải ở trạng thái `running` và thiết bị Edge phải thuộc đúng xe của chuyến:

```cmd
python scripts\18_run_edge_client.py --trip-id <ID_CHUYEN> --simulate-gps --interval 10
```

Không dùng tọa độ mô phỏng này như dữ liệu đo thực tế. Có thể tắt tiếng báo Edge
bằng `set DMS_EDGE_AUDIO_ALERT=false` trước khi chạy `20_run_edge_simulator.py`.

## 4. Backup MySQL

```cmd
set DMS_DATABASE_URL=mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms
python scripts\15_backup_mysql.py
```

SQLite chỉ được dùng trong unit test tạm thời, không thuộc kiến trúc hoặc demo.
# Âm thanh cảnh báo tiếng Việt

Edge ưu tiên các tệp WAV tiếng Việt đóng gói trong `backend/app/services`. Mở `START_EDGE_DEMO.vbs` và bấm **Thử giọng Việt** trước khi demo. Nếu thiếu tệp WAV, chương trình thử giọng `vi-VN` của Windows rồi mới chuyển sang tiếng bíp.

# Xuất CSV

Trong **Lịch sử cảnh báo**, chọn phiên, loại dòng, cột sắp xếp và chiều sắp xếp trước khi bấm **Tải CSV**. Tệp có một hàng tiêu đề cố định và dấu phân cách `;`, nên các cột mở đúng trong Excel; có thể bật Data → Filter để lọc thêm mà không cần tệp `.xlsx`.
