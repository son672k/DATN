# DMS Mobile

Một ứng dụng Flutter Android cho hai vai trò, không phải thiết bị camera AI.

Chức năng hiện tại:

- Supervisor xem đội tài xế, phương tiện, chuyến/GPS, toàn bộ cảnh báo và cập nhật xử lý.
- Tài xế chỉ xem hồ sơ, chuyến/GPS, cảnh báo, ảnh và clip thuộc chính mình.
- JWT được giữ trong secure storage; backend kiểm tra quyền ở mọi API dữ liệu tài xế.
- Ứng dụng kiểm tra cảnh báo mới mỗi 15 giây và hiện Android local notification.
- Kéo xuống để đồng bộ lại dữ liệu MySQL qua FastAPI.

Ứng dụng không xin quyền camera, không gửi frame và không tạo phiên giám sát.

```cmd
cd mobile_app
flutter pub get
flutter run
```

Android Emulator dùng `http://10.0.2.2:8000`. Điện thoại thật dùng IP LAN của máy chạy backend.

Thông báo hiện tại phục vụ demo khi ứng dụng còn hoạt động. Push notification
khi ứng dụng bị tắt sẽ được nối Firebase/AWS ở giai đoạn triển khai cloud cuối.
