# DMS Backend

Backend FastAPI sử dụng MySQL làm cơ sở dữ liệu runtime duy nhất.

Các nhóm API chính:

- Xác thực JWT và phân quyền supervisor/tài xế.
- API `/api/driver/me/*` chống truy cập chéo dữ liệu giữa các tài xế.
- Quản lý tài xế, phương tiện, tuyến và chuyến; mỗi chuyến gắn trực tiếp tài xế với xe.
- Quản lý chuyến đi và GPS; hỗ trợ xóa toàn bộ chuyến đã hoàn thành/đã hủy cùng dữ liệu liên quan.
- Điều khiển Edge Simulator từ cabin camera/video và phát cảnh báo âm thanh cục bộ.
- Lưu metric AI, cảnh báo, snapshot và báo cáo.

Không còn API nhận camera/frame từ điện thoại. Mobile App chỉ gọi API quản lý.

Khởi tạo và kiểm tra:

```cmd
set DMS_DATABASE_URL=mysql://dms_app:change-me-before-deploying@127.0.0.1:3306/dms
python scripts\13_check_mysql.py
```

Unit test có thể dùng SQLite tạm trong thư mục hệ thống để tránh thay đổi dữ liệu MySQL thật. SQLite không được dùng khi chạy ứng dụng.
