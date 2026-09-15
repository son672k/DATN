import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:video_player/video_player.dart';

import 'services/dms_api_client.dart';

final _notifications = FlutterLocalNotificationsPlugin();

String _eventLabel(Object? value) {
  const labels = {
    'drowsy': 'Nghi ngờ buồn ngủ theo chuỗi',
    'high_perclos': 'Tỷ lệ nhắm mắt cao (PERCLOS ước lượng)',
    'phone': 'Phát hiện điện thoại',
    'cigarette': 'Phát hiện thuốc lá',
    'yawn': 'Nghi ngờ ngáp kéo dài',
    'microsleep': 'Nghi ngờ nhắm mắt kéo dài',
    'distraction': 'Nghi ngờ mất tập trung thị giác',
    'no_seatbelt': 'Nghi ngờ không thắt dây an toàn',
  };
  final key = value?.toString() ?? '';
  return labels[key] ?? (key.isEmpty ? 'Cảnh báo' : key);
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _notifications.initialize(
    const InitializationSettings(
      android: AndroidInitializationSettings('@mipmap/ic_launcher'),
    ),
  );
  runApp(const DmsManagerApp());
}

class DmsManagerApp extends StatelessWidget {
  const DmsManagerApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
      title: 'DMS Mobile',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff087f6b),
          primary: const Color(0xff087f6b),
          secondary: const Color(0xffe59b24),
          surface: const Color(0xfff8fbfa),
        ),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xffeaf2ef),
        textTheme: const TextTheme(
          bodyLarge: TextStyle(fontSize: 16, height: 1.5),
          bodyMedium: TextStyle(fontSize: 15, height: 1.5),
          bodySmall: TextStyle(fontSize: 13, height: 1.5),
          titleLarge: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              color: Color(0xff102a28)),
          titleMedium: TextStyle(fontSize: 17, fontWeight: FontWeight.w600),
          labelLarge: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xff063f38),
          foregroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 1,
          surfaceTintColor: Color(0xff063f38),
        ),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          margin: const EdgeInsets.only(bottom: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
            side: const BorderSide(color: Color(0xffdce8e4)),
          ),
        ),
        navigationBarTheme: const NavigationBarThemeData(
          height: 72,
          backgroundColor: Color(0xff063f38),
          indicatorColor: Color(0xff18a98e),
          iconTheme: WidgetStatePropertyAll(
            IconThemeData(color: Color(0xffd7f3eb)),
          ),
          labelTextStyle: WidgetStatePropertyAll(
            TextStyle(
                fontSize: 12, fontWeight: FontWeight.w700, color: Colors.white),
          ),
        ),
        dialogTheme: DialogThemeData(
          backgroundColor: Colors.white,
          surfaceTintColor: Colors.white,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
        ),
        snackBarTheme: const SnackBarThemeData(
          behavior: SnackBarBehavior.floating,
          showCloseIcon: true,
        ),
        listTileTheme: const ListTileThemeData(
          contentPadding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          minVerticalPadding: 12,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
          border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(14),
              borderSide: const BorderSide(color: Color(0xffc8d9d3))),
          enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(14),
              borderSide: const BorderSide(color: Color(0xffc8d9d3))),
        ),
        filledButtonTheme: FilledButtonThemeData(
            style: FilledButton.styleFrom(
                minimumSize: const Size(48, 48),
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 14))),
        outlinedButtonTheme: OutlinedButtonThemeData(
            style: OutlinedButton.styleFrom(minimumSize: const Size(48, 48))),
        textButtonTheme: TextButtonThemeData(
            style: TextButton.styleFrom(minimumSize: const Size(48, 48))),
      ),
      home: const ManagerHome());
}

class ManagerHome extends StatefulWidget {
  const ManagerHome({super.key});
  @override
  State<ManagerHome> createState() => _ManagerHomeState();
}

class _ManagerHomeState extends State<ManagerHome> {
  static const _storage = FlutterSecureStorage();
  final _baseUrl = TextEditingController(text: 'http://10.0.2.2:8000');
  final _username = TextEditingController();
  final _password = TextEditingController();
  String? _token, _error;
  Map<String, dynamic>? _user;
  Timer? _alertTimer;
  bool _pollingAlerts = false;
  int _lastAlertId = 0;
  bool _busy = true;
  int _tab = 0;
  int? _selectedDriverId;
  String? _selectedDriverName;
  List<Map<String, dynamic>> _fleet = [],
      _alerts = [],
      _vehicles = [],
      _trips = [];
  DmsApiClient get _api => DmsApiClient(_baseUrl.text.trim(), token: _token);
  bool get _isSupervisor => _user?['role'] == 'supervisor';

  @override
  void initState() {
    super.initState();
    _restore();
  }

  @override
  void dispose() {
    _alertTimer?.cancel();
    _baseUrl.dispose();
    _username.dispose();
    _password.dispose();
    super.dispose();
  }

  Future<void> _restore() async {
    final prefs = await SharedPreferences.getInstance();
    _baseUrl.text = prefs.getString('api_base_url') ?? _baseUrl.text;
    _token = await _storage.read(key: 'auth_token') ??
        await _storage.read(key: 'supervisor_token');
    if (_token != null) {
      try {
        _user = await _api.me();
        await _refresh();
        await _startAlertPolling();
      } catch (_) {
        _token = null;
        _user = null;
      }
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _login() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('api_base_url', _baseUrl.text.trim());
      final result = await DmsApiClient(_baseUrl.text.trim())
          .login(_username.text.trim().toLowerCase(), _password.text);
      _token = result['access_token'] as String;
      _user = Map<String, dynamic>.from(result['user'] as Map);
      await _storage.write(key: 'auth_token', value: _token);
      await _storage.delete(key: 'supervisor_token');
      await _refresh();
      await _startAlertPolling();
    } catch (error) {
      _token = null;
      _user = null;
      _error = error.toString();
    }
    if (mounted) setState(() => _busy = false);
  }

  Future<void> _refresh() async {
    final data = _isSupervisor
        ? await Future.wait(
            [_api.fleet(), _api.alerts(), _api.vehicles(), _api.trips()])
        : await Future.wait([_api.driverAlerts(), _api.driverTrips()]);
    if (mounted) {
      setState(() {
        if (_isSupervisor) {
          _fleet = data[0];
          _alerts = data[1];
          _vehicles = data[2];
          _trips = data[3];
        } else {
          _fleet = [];
          _alerts = data[0];
          _vehicles = [];
          _trips = data[1];
        }
        _error = null;
      });
    }
  }

  Future<void> _logout() async {
    _alertTimer?.cancel();
    await _storage.delete(key: 'auth_token');
    await _storage.delete(key: 'supervisor_token');
    setState(() {
      _token = null;
      _user = null;
      _tab = 0;
      _fleet = [];
      _alerts = [];
      _vehicles = [];
      _trips = [];
      _lastAlertId = 0;
      _selectedDriverId = null;
      _selectedDriverName = null;
    });
  }

  int _greatestAlertId(List<Map<String, dynamic>> alerts) => alerts.fold<int>(
      0,
      (current, alert) => (alert['id'] as num?)?.toInt() != null &&
              (alert['id'] as num).toInt() > current
          ? (alert['id'] as num).toInt()
          : current);

  Future<void> _startAlertPolling() async {
    _alertTimer?.cancel();
    _lastAlertId = _greatestAlertId(_alerts);
    if (Platform.isAndroid) {
      await _notifications
          .resolvePlatformSpecificImplementation<
              AndroidFlutterLocalNotificationsPlugin>()
          ?.requestNotificationsPermission();
    }
    _alertTimer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _pollAlerts(),
    );
  }

  Future<void> _pollAlerts() async {
    if (_token == null || _pollingAlerts) return;
    _pollingAlerts = true;
    try {
      final latest =
          _isSupervisor ? await _api.alerts() : await _api.driverAlerts();
      final newAlerts = latest
          .where((alert) => (alert['id'] as num? ?? 0).toInt() > _lastAlertId)
          .toList();
      if (newAlerts.isNotEmpty) {
        newAlerts.sort((a, b) =>
            (a['id'] as num).toInt().compareTo((b['id'] as num).toInt()));
        final alert = newAlerts.last;
        await _notifications.show(
          (alert['id'] as num).toInt(),
          'Cảnh báo DMS: ${_eventLabel(alert['event_type'])}',
          '${alert['driver_name'] ?? 'Tài xế'} · Risk ${alert['risk_score'] ?? 0}/100',
          const NotificationDetails(
            android: AndroidNotificationDetails(
              'dms_safety_alerts',
              'Cảnh báo an toàn DMS',
              channelDescription:
                  'Thông báo sự kiện mất an toàn từ hệ thống DMS',
              importance: Importance.high,
              priority: Priority.high,
            ),
          ),
        );
      }
      _lastAlertId = _greatestAlertId(latest);
      if (mounted) setState(() => _alerts = latest);
    } catch (_) {
      // Mất mạng tạm thời không đăng xuất người dùng; lần polling sau sẽ thử lại.
    } finally {
      _pollingAlerts = false;
    }
  }

  String get _backendHost {
    final value = _baseUrl.text.trim();
    return value.replaceFirst(RegExp(r'^https?://'), '');
  }

  Future<void> _showBackendDialog() async {
    final draft = TextEditingController(text: _baseUrl.text.trim());
    final saved = await showDialog<String>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        icon: const Icon(Icons.dns_outlined),
        title: const Text('Kết nối máy chủ'),
        content: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 420),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Nhập địa chỉ FastAPI. Điện thoại thật dùng địa chỉ IPv4 LAN của máy tính.',
              ),
              const SizedBox(height: 16),
              TextField(
                controller: draft,
                keyboardType: TextInputType.url,
                autocorrect: false,
                decoration: const InputDecoration(
                  labelText: 'Địa chỉ backend',
                  hintText: 'http://192.168.1.10:8000',
                  prefixIcon: Icon(Icons.link),
                ),
              ),
              const SizedBox(height: 10),
              const Text(
                'Android Emulator: http://10.0.2.2:8000',
                style: TextStyle(fontSize: 12, color: Color(0xff607773)),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Hủy'),
          ),
          FilledButton(
            onPressed: () {
              var value = draft.text.trim();
              if (value.isNotEmpty &&
                  !value.startsWith('http://') &&
                  !value.startsWith('https://')) {
                value = 'http://$value';
              }
              Navigator.pop(dialogContext, value);
            },
            child: const Text('Lưu địa chỉ'),
          ),
        ],
      ),
    );
    draft.dispose();
    if (saved == null || saved.isEmpty) return;
    _baseUrl.text = saved.replaceAll(RegExp(r'/$'), '');
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_base_url', _baseUrl.text);
    if (mounted) {
      setState(() {});
      _feedback('Đã lưu địa chỉ máy chủ.');
    }
  }

  Widget _buildLogin() => Scaffold(
        body: SafeArea(
          child: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Color(0xffe5f3ee), Color(0xfff7efe1)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 460),
                  child: Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: const Color(0xeef8fbfa),
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: const Color(0xffc8ddd6)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Color(0xff063f38), Color(0xff0c8c75)],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(28),
                            boxShadow: const [
                              BoxShadow(
                                color: Color(0x28063f38),
                                blurRadius: 28,
                                offset: Offset(0, 14),
                              )
                            ],
                          ),
                          child: const Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              CircleAvatar(
                                radius: 26,
                                backgroundColor: Color(0x26ffffff),
                                child: Icon(Icons.shield_outlined,
                                    color: Colors.white, size: 29),
                              ),
                              SizedBox(height: 22),
                              Text('DMS Control',
                                  style: TextStyle(
                                      color: Colors.white,
                                      fontSize: 28,
                                      fontWeight: FontWeight.w800)),
                              SizedBox(height: 7),
                              Text(
                                'Theo dõi chuyến đi và tiếp nhận cảnh báo an toàn theo vai trò.',
                                style: TextStyle(
                                    color: Color(0xffd7f3eb),
                                    fontSize: 15,
                                    height: 1.45),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 28),
                        Text('Đăng nhập hệ thống',
                            style: Theme.of(context).textTheme.titleLarge),
                        const SizedBox(height: 6),
                        Row(
                          children: [
                            const Icon(Icons.cloud_done_outlined,
                                size: 17, color: Color(0xff087f6b)),
                            const SizedBox(width: 7),
                            Expanded(
                              child: Text(
                                _backendHost,
                                overflow: TextOverflow.ellipsis,
                                style:
                                    const TextStyle(color: Color(0xff607773)),
                              ),
                            ),
                            TextButton.icon(
                              onPressed: _showBackendDialog,
                              icon: const Icon(Icons.tune, size: 18),
                              label: const Text('Thiết lập'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 14),
                        TextField(
                          controller: _username,
                          textInputAction: TextInputAction.next,
                          decoration: const InputDecoration(
                            labelText: 'Tên đăng nhập',
                            prefixIcon: Icon(Icons.person_outline),
                          ),
                        ),
                        const SizedBox(height: 14),
                        TextField(
                          controller: _password,
                          obscureText: true,
                          onSubmitted: (_) => _login(),
                          decoration: const InputDecoration(
                            labelText: 'Mật khẩu',
                            prefixIcon: Icon(Icons.lock_outline),
                          ),
                        ),
                        const SizedBox(height: 20),
                        FilledButton.icon(
                          onPressed: _login,
                          icon: const Icon(Icons.login),
                          label: const Text('Đăng nhập'),
                        ),
                        if (_error != null)
                          Container(
                            margin: const EdgeInsets.only(top: 16),
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: const Color(0xffffeeee),
                              borderRadius: BorderRadius.circular(14),
                            ),
                            child: Text(_error!,
                                style:
                                    const TextStyle(color: Color(0xffa92d2d))),
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    if (_busy) {
      return const Scaffold(
          body: Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
        CircularProgressIndicator(),
        SizedBox(height: 16),
        Text('Đang kết nối hệ thống…')
      ])));
    }
    if (_token == null) {
      return _buildLogin();
    }
    final supervisorPages = [
      _driverItems(),
      _alertItems(),
      _items('Phương tiện', _vehicles, (x) => x['plate_number'] ?? 'Xe',
          (x) => '${x['vehicle_type'] ?? ''} · ${x['status'] ?? ''}'),
      _tripItems(),
    ];
    final driverPages = [_driverHome(), _alertItems(), _tripItems()];
    final pages = _isSupervisor ? supervisorPages : driverPages;
    final destinations = _isSupervisor
        ? const [
            NavigationDestination(icon: Icon(Icons.people), label: 'Tài xế'),
            NavigationDestination(
                icon: Icon(Icons.warning_amber), label: 'Cảnh báo'),
            NavigationDestination(
                icon: Icon(Icons.directions_car), label: 'Xe'),
            NavigationDestination(icon: Icon(Icons.route), label: 'Chuyến'),
          ]
        : const [
            NavigationDestination(icon: Icon(Icons.person), label: 'Cá nhân'),
            NavigationDestination(
                icon: Icon(Icons.warning_amber), label: 'Vi phạm'),
            NavigationDestination(icon: Icon(Icons.route), label: 'Chuyến'),
          ];
    return Scaffold(
        appBar: AppBar(
            leadingWidth: 62,
            leading: Padding(
              padding: const EdgeInsets.only(left: 16, top: 8, bottom: 8),
              child: Container(
                decoration: BoxDecoration(
                  color: const Color(0xffd8f1e8),
                  borderRadius: BorderRadius.circular(13),
                ),
                child:
                    const Icon(Icons.shield_outlined, color: Color(0xff087f6b)),
              ),
            ),
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                    _isSupervisor ? 'Trung tâm điều hành' : 'Không gian tài xế',
                    style: const TextStyle(
                        fontSize: 17, fontWeight: FontWeight.w800)),
                Text(_user?['display_name']?.toString() ?? '',
                    style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w400,
                        color: Color(0xffbfe9dd))),
              ],
            ),
            actions: [
              IconButton(
                  tooltip: 'Kết nối máy chủ',
                  onPressed: _showBackendDialog,
                  icon: const Icon(Icons.dns_outlined)),
              IconButton(
                  tooltip: 'Làm mới',
                  onPressed: _manualRefresh,
                  icon: const Icon(Icons.refresh)),
              PopupMenuButton<String>(
                tooltip: 'Tài khoản',
                onSelected: (value) {
                  if (value == 'logout') _logout();
                },
                itemBuilder: (_) => const [
                  PopupMenuItem(
                    value: 'logout',
                    child: ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: Icon(Icons.logout),
                      title: Text('Đăng xuất'),
                    ),
                  )
                ],
              ),
            ]),
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xffe8f3ef), Color(0xfff5efe4)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: SafeArea(
            top: false,
            child:
                RefreshIndicator(onRefresh: _manualRefresh, child: pages[_tab]),
          ),
        ),
        bottomNavigationBar: NavigationBar(
            selectedIndex: _tab,
            onDestinationSelected: (v) => setState(() => _tab = v),
            destinations: destinations));
  }

  Widget _items(
          String title,
          List<Map<String, dynamic>> items,
          String Function(Map<String, dynamic>) name,
          String Function(Map<String, dynamic>) detail) =>
      ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
          children: [
            _pageHeading(
              title,
              title == 'Tài xế'
                  ? 'Theo dõi hồ sơ và số cảnh báo cần xử lý'
                  : 'Danh sách phương tiện đang được quản lý',
              count: items.length,
            ),
            const SizedBox(height: 18),
            if (items.isEmpty)
              _emptyCard(Icons.inbox_outlined, 'Chưa có dữ liệu'),
            ...items.map((item) {
              final icon = title == 'Tài xế'
                  ? Icons.person_outline
                  : Icons.directions_car_outlined;
              final accent = title == 'Tài xế'
                  ? const Color(0xff167c70)
                  : const Color(0xff315f9c);
              return Container(
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      accent.withValues(alpha: .13),
                      const Color(0xfffbfdfc)
                    ],
                  ),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: accent.withValues(alpha: .24)),
                  boxShadow: const [
                    BoxShadow(
                        color: Color(0x10063f38),
                        blurRadius: 12,
                        offset: Offset(0, 5))
                  ],
                ),
                child: ListTile(
                  leading: _leadingIcon(icon),
                  title: Text(name(item),
                      style: const TextStyle(fontWeight: FontWeight.w700)),
                  subtitle: Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(detail(item)),
                  ),
                  trailing: Icon(Icons.chevron_right, color: accent),
                ),
              );
            }),
          ]);

  Widget _driverItems() => ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
        children: [
          _pageHeading(
            'Tài xế',
            'Chọn tài xế để lọc cảnh báo hoặc chuyến đi',
            count: _fleet.length,
          ),
          const SizedBox(height: 18),
          if (_fleet.isEmpty)
            _emptyCard(Icons.people_outline, 'Chưa có tài xế'),
          ..._fleet.map((item) {
            final driver = Map<String, dynamic>.from(item['driver'] as Map);
            final driverId = (driver['id'] as num).toInt();
            final selected = driverId == _selectedDriverId;
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: selected
                      ? [const Color(0xffc9eee4), const Color(0xfff8fbfa)]
                      : [const Color(0xffe4f3ef), const Color(0xfffbfdfc)],
                ),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: selected
                      ? const Color(0xff087f6b)
                      : const Color(0xffbcd9d1),
                  width: selected ? 1.5 : 1,
                ),
              ),
              child: ListTile(
                leading: _leadingIcon(Icons.person_outline),
                title: Text(driver['display_name']?.toString() ?? 'Tài xế',
                    style: const TextStyle(fontWeight: FontWeight.w800)),
                subtitle: Padding(
                  padding: const EdgeInsets.only(top: 5),
                  child: Text(
                    '${driver['external_id'] ?? 'Chưa có mã'} · Phiên: ${item['total_sessions'] ?? 0}\n'
                    'Cảnh báo mới: ${item['review_counts']?['new'] ?? 0}',
                  ),
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => _showDriverActions(item),
              ),
            );
          }),
        ],
      );

  Future<void> _showDriverActions(Map<String, dynamic> item) async {
    final driver = Map<String, dynamic>.from(item['driver'] as Map);
    final driverId = (driver['id'] as num).toInt();
    final driverName = driver['display_name']?.toString() ?? 'Tài xế';
    await showModalBottomSheet<void>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      backgroundColor: const Color(0xfff8fbfa),
      builder: (sheetContext) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(22, 4, 22, 24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(children: [
                _leadingIcon(Icons.person_outline),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(driverName,
                          style: Theme.of(context).textTheme.titleLarge),
                      Text(
                          '${driver['external_id'] ?? 'Chưa có mã'} · ${driver['phone'] ?? 'Chưa có số điện thoại'}',
                          style: const TextStyle(color: Color(0xff607773))),
                    ],
                  ),
                ),
              ]),
              const SizedBox(height: 18),
              Row(children: [
                Expanded(
                    child: _summaryCard(
                        Icons.video_camera_front_outlined,
                        '${item['total_sessions'] ?? 0}',
                        'Phiên',
                        const Color(0xff315f9c))),
                const SizedBox(width: 12),
                Expanded(
                    child: _summaryCard(
                        Icons.warning_amber_rounded,
                        '${item['review_counts']?['new'] ?? 0}',
                        'Cảnh báo mới',
                        const Color(0xffd97706))),
              ]),
              const SizedBox(height: 16),
              FilledButton.icon(
                onPressed: () {
                  Navigator.pop(sheetContext);
                  setState(() {
                    _selectedDriverId = driverId;
                    _selectedDriverName = driverName;
                    _tab = 1;
                  });
                },
                icon: const Icon(Icons.filter_alt_outlined),
                label: const Text('Xem cảnh báo của tài xế'),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: () {
                  Navigator.pop(sheetContext);
                  setState(() {
                    _selectedDriverId = driverId;
                    _selectedDriverName = driverName;
                    _tab = 3;
                  });
                },
                icon: const Icon(Icons.route_outlined),
                label: const Text('Xem chuyến đi của tài xế'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _driverFilterBar() {
    if (!_isSupervisor || _fleet.isEmpty) return const SizedBox.shrink();
    return SizedBox(
      height: 44,
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: [
          ChoiceChip(
            label: const Text('Tất cả tài xế'),
            selected: _selectedDriverId == null,
            onSelected: (_) => setState(() {
              _selectedDriverId = null;
              _selectedDriverName = null;
            }),
          ),
          const SizedBox(width: 8),
          ..._fleet.map((item) {
            final driver = item['driver'] as Map;
            final id = (driver['id'] as num).toInt();
            final name = driver['display_name']?.toString() ?? 'Tài xế';
            return Padding(
              padding: const EdgeInsets.only(right: 8),
              child: ChoiceChip(
                label: Text(name),
                selected: _selectedDriverId == id,
                onSelected: (_) => setState(() {
                  _selectedDriverId = id;
                  _selectedDriverName = name;
                }),
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _pageHeading(String title, String subtitle, {required int count}) =>
      Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
              colors: [Color(0xff063f38), Color(0xff12836f)]),
          borderRadius: BorderRadius.circular(20),
          boxShadow: const [
            BoxShadow(
                color: Color(0x24063f38), blurRadius: 18, offset: Offset(0, 8))
          ],
        ),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(title,
                  style: Theme.of(context)
                      .textTheme
                      .titleLarge
                      ?.copyWith(color: Colors.white)),
              const SizedBox(height: 4),
              Text(subtitle,
                  style:
                      const TextStyle(color: Color(0xffc6ebe1), fontSize: 13)),
            ]),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
            decoration: BoxDecoration(
              color: const Color(0x33ffffff),
              borderRadius: BorderRadius.circular(99),
            ),
            child: Text('$count',
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.w800)),
          ),
        ]),
      );

  Widget _leadingIcon(IconData icon, {Color color = const Color(0xff087f6b)}) =>
      Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: color.withValues(alpha: .10),
          borderRadius: BorderRadius.circular(13),
        ),
        child: Icon(icon, color: color),
      );

  Widget _emptyCard(IconData icon, String text) => Card(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 30),
          child: Column(children: [
            Icon(icon, size: 34, color: const Color(0xff88a09b)),
            const SizedBox(height: 10),
            Text(text, style: const TextStyle(color: Color(0xff607773))),
          ]),
        ),
      );

  Widget _driverHome() {
    final driver = _user?['driver'] as Map<String, dynamic>?;
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
      children: [
        _pageHeading('Tổng quan cá nhân',
            'Dữ liệu an toàn thuộc tài khoản đang đăng nhập',
            count: _trips.length),
        const SizedBox(height: 18),
        Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xff063f38), Color(0xff0c8c75)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(22),
          ),
          child: Row(children: [
            const CircleAvatar(
              radius: 29,
              backgroundColor: Color(0x26ffffff),
              child: Icon(Icons.person_outline, color: Colors.white, size: 30),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(_user?['display_name']?.toString() ?? 'Tài xế',
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  Text('Mã tài xế: ${driver?['external_id'] ?? '—'}',
                      style: const TextStyle(color: Color(0xffd7f3eb))),
                  Text('@${_user?['username'] ?? '—'}',
                      style: const TextStyle(color: Color(0xffaee2d4))),
                ],
              ),
            ),
          ]),
        ),
        const SizedBox(height: 14),
        Row(children: [
          Expanded(
              child: _summaryCard(Icons.route_outlined, '${_trips.length}',
                  'Chuyến đi', const Color(0xff087f6b))),
          const SizedBox(width: 12),
          Expanded(
              child: _summaryCard(Icons.warning_amber_rounded,
                  '${_alerts.length}', 'Cảnh báo', const Color(0xffd97706))),
        ]),
        const SizedBox(height: 14),
        const Card(
          child: ListTile(
            leading: Icon(Icons.info_outline, color: Color(0xff087f6b)),
            title: Text('Quyền riêng tư và xử lý AI',
                style: TextStyle(fontWeight: FontWeight.w700)),
            subtitle: Text(
                'Ứng dụng chỉ hiển thị dữ liệu của tài khoản này. Camera và mô hình AI chạy trên thiết bị Edge trong xe.'),
          ),
        ),
      ],
    );
  }

  Widget _summaryCard(IconData icon, String value, String label, Color color) =>
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xffdce8e4)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, color: color),
          const SizedBox(height: 12),
          Text(value,
              style:
                  const TextStyle(fontSize: 25, fontWeight: FontWeight.w800)),
          Text(label, style: const TextStyle(color: Color(0xff607773))),
        ]),
      );

  Widget _alertItems() {
    final visibleAlerts = _selectedDriverId == null
        ? _alerts
        : _alerts
            .where((item) =>
                (item['driver_id'] as num?)?.toInt() == _selectedDriverId)
            .toList();
    return ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
        children: [
          _pageHeading(
              'Cảnh báo',
              _selectedDriverName == null
                  ? 'Nhấn một mục để xem bằng chứng và xử lý'
                  : 'Đang lọc theo $_selectedDriverName',
              count: visibleAlerts.length),
          if (_isSupervisor) ...[
            const SizedBox(height: 14),
            _driverFilterBar(),
          ],
          const SizedBox(height: 18),
          if (visibleAlerts.isEmpty)
            _emptyCard(
                Icons.verified_user_outlined,
                _selectedDriverId == null
                    ? 'Chưa có cảnh báo'
                    : 'Tài xế này chưa có cảnh báo'),
          ...visibleAlerts.map((item) => Card(
                  child: ListTile(
                leading: _leadingIcon(Icons.warning_amber_rounded,
                    color: (item['risk_score'] as num? ?? 0) >= 70
                        ? const Color(0xffc53b32)
                        : const Color(0xffd97706)),
                title: Text(_eventLabel(item['event_type']),
                    style: const TextStyle(fontWeight: FontWeight.w700)),
                subtitle: Padding(
                  padding: const EdgeInsets.only(top: 5),
                  child: Text(
                      '${item['driver_name'] ?? 'Chưa gán'} · Rủi ro ${item['risk_score'] ?? 0}/100\n${item['review_status'] ?? 'new'}'),
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => _showAlert(item),
              ))),
        ]);
  }

  Future<void> _showAlert(Map<String, dynamic> alert) async {
    Uint8List? snapshot;
    String? loadError;
    if (alert['snapshot_path'] != null) {
      try {
        snapshot = Uint8List.fromList(await _api.snapshot(alert['id'] as int));
      } catch (error) {
        loadError = error.toString();
      }
    }
    if (!mounted) return;
    await showDialog<void>(
        context: context,
        builder: (dialogContext) => AlertDialog(
              title: Text(_eventLabel(alert['event_type'])),
              content: SingleChildScrollView(
                  child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Tài xế: ${alert['driver_name'] ?? 'Chưa gán'}'),
                  Text('Điểm rủi ro: ${alert['risk_score'] ?? 0}/100'),
                  Text('Trạng thái: ${alert['review_status'] ?? 'new'}'),
                  Text('Mức cảnh báo: ${alert['severity'] ?? 'Chưa xác định'}'),
                  Text(alert['latitude'] == null
                      ? 'GPS sự kiện: chưa có điểm hợp lệ'
                      : 'GPS sự kiện: ${alert['latitude']}, ${alert['longitude']}'),
                  if (alert['gps_source'] != null)
                    Text('Nguồn GPS: ${alert['gps_source']}'),
                  const SizedBox(height: 12),
                  if (snapshot != null)
                    ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(snapshot)),
                  if (loadError != null)
                    Text(loadError, style: const TextStyle(color: Colors.red)),
                  if (alert['clip_path'] != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: OutlinedButton.icon(
                        onPressed: () => _playClip(alert['id'] as int),
                        icon: const Icon(Icons.play_circle_outline),
                        label: const Text('Phát clip bằng chứng'),
                      ),
                    ),
                ],
              )),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(dialogContext),
                    child: const Text('Đóng')),
                if (_isSupervisor && alert['review_status'] == 'new')
                  TextButton(
                      onPressed: () async {
                        Navigator.pop(dialogContext);
                        await _updateAlert(alert, 'acknowledged');
                      },
                      child: const Text('Tiếp nhận')),
                if (_isSupervisor && alert['review_status'] != 'resolved')
                  FilledButton(
                      onPressed: () async {
                        Navigator.pop(dialogContext);
                        await _updateAlert(alert, 'resolved');
                      },
                      child: const Text('Đã xử lý')),
              ],
            ));
  }

  bool _actionPending = false;

  void _feedback(String message) {
    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(SnackBar(
      content: Text(message),
      duration: const Duration(seconds: 5),
      behavior: SnackBarBehavior.floating,
      action: SnackBarAction(
          label: 'Đóng', onPressed: () => messenger.hideCurrentSnackBar()),
    ));
  }

  Future<void> _manualRefresh() async {
    try {
      await _refresh();
      _feedback('Đã cập nhật dữ liệu mới nhất.');
    } catch (error) {
      _feedback('Không cập nhật được dữ liệu: $error');
    }
  }

  Future<void> _updateAlert(Map<String, dynamic> alert, String status) async {
    if (_actionPending) return;
    _actionPending = true;
    _feedback('Đang xử lý…');
    try {
      await _api.reviewAlert(alert['id'] as int, status, null);
      _feedback(status == 'resolved'
          ? 'Đã đánh dấu cảnh báo được xử lý.'
          : status == 'acknowledged'
              ? 'Đã tiếp nhận cảnh báo.'
              : 'Đã mở lại cảnh báo.');
      try {
        await _refresh();
      } catch (_) {
        _feedback(
            'Đã lưu thay đổi, nhưng chưa tải lại được danh sách. Kéo xuống để làm mới.');
      }
    } catch (error) {
      _feedback('Không thực hiện được thao tác: $error');
    } finally {
      _actionPending = false;
    }
  }

  Future<void> _playClip(int eventId) async {
    try {
      final bytes = await _api.clip(eventId);
      final directory = await getTemporaryDirectory();
      final file = File(
          '${directory.path}${Platform.pathSeparator}dms_event_$eventId.mp4');
      await file.writeAsBytes(bytes, flush: true);
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (_) => EvidenceVideoDialog(file: file),
      );
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Không phát được clip: $error')));
      }
    }
  }

  Widget _tripItems() {
    final visibleTrips = _selectedDriverId == null
        ? _trips
        : _trips
            .where((trip) =>
                (trip['driver_id'] as num?)?.toInt() == _selectedDriverId)
            .toList();
    return ListView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
        children: [
          _pageHeading(
              'Chuyến đi',
              _selectedDriverName == null
                  ? 'Theo dõi trạng thái và vị trí gần nhất'
                  : 'Đang lọc theo $_selectedDriverName',
              count: visibleTrips.length),
          if (_isSupervisor) ...[
            const SizedBox(height: 14),
            _driverFilterBar(),
          ],
          const SizedBox(height: 18),
          if (visibleTrips.isEmpty)
            _emptyCard(
                Icons.route_outlined,
                _selectedDriverId == null
                    ? 'Chưa có chuyến đi'
                    : 'Tài xế này chưa có chuyến đi'),
          ...visibleTrips.map((trip) => Card(
                  child: ListTile(
                leading: _leadingIcon(Icons.route_outlined),
                title: Text(trip['trip_code']?.toString() ?? 'Chuyến',
                    style: const TextStyle(fontWeight: FontWeight.w700)),
                subtitle: Padding(
                  padding: const EdgeInsets.only(top: 5),
                  child: Text(
                      '${trip['driver_name'] ?? 'Chưa gán'} · ${trip['plate_number'] ?? 'Chưa gán xe'}\n${trip['route_name'] ?? 'Chưa chọn tuyến'} · ${trip['status'] ?? ''}'),
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () => _showTrip(trip),
              ))),
        ]);
  }

  Future<void> _showTrip(Map<String, dynamic> trip) async {
    List<Map<String, dynamic>> locations = [];
    String? loadError;
    try {
      locations =
          await _api.tripLocations(trip['id'] as int, driver: !_isSupervisor);
    } catch (error) {
      loadError = error.toString();
    }
    if (!mounted) return;
    final latest = locations.isNotEmpty ? locations.last : null;
    await showDialog<void>(
        context: context,
        builder: (dialogContext) => AlertDialog(
              title: Text('Chuyến ${trip['trip_code'] ?? ''}'),
              content: SingleChildScrollView(
                  child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Tài xế: ${trip['driver_name'] ?? 'Chưa gán'}'),
                  Text('Phương tiện: ${trip['plate_number'] ?? 'Chưa gán'}'),
                  Text('Tuyến: ${trip['route_name'] ?? 'Chưa chọn'}'),
                  Text('Trạng thái: ${trip['status'] ?? '—'}'),
                  const Divider(height: 24),
                  Text('GPS (${locations.length} điểm)',
                      style: Theme.of(context).textTheme.titleMedium),
                  if (latest != null) ...[
                    Text(
                        'Vị trí mới nhất: ${latest['latitude']}, ${latest['longitude']}'),
                    Text('Tốc độ: ${latest['speed_kph'] ?? '—'} km/h'),
                    Text('Thời gian: ${latest['recorded_at'] ?? '—'}'),
                  ] else
                    const Text('Chưa có dữ liệu GPS.'),
                  if (loadError != null)
                    Text(loadError, style: const TextStyle(color: Colors.red)),
                ],
              )),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(dialogContext),
                    child: const Text('Đóng')),
                if (_isSupervisor && trip['status'] == 'planned')
                  FilledButton(
                      onPressed: () async {
                        Navigator.pop(dialogContext);
                        await _changeTripStatus(trip, 'running');
                      },
                      child: const Text('Bắt đầu chuyến')),
                if (_isSupervisor && trip['status'] == 'running')
                  FilledButton(
                      onPressed: () async {
                        Navigator.pop(dialogContext);
                        await _changeTripStatus(trip, 'completed');
                      },
                      child: const Text('Hoàn thành')),
                if (_isSupervisor &&
                    (trip['status'] == 'planned' ||
                        trip['status'] == 'running'))
                  TextButton(
                      onPressed: () async {
                        Navigator.pop(dialogContext);
                        await _changeTripStatus(trip, 'cancelled');
                      },
                      child: const Text('Hủy chuyến')),
              ],
            ));
  }

  Future<void> _changeTripStatus(
      Map<String, dynamic> trip, String status) async {
    if (_actionPending) return;
    _actionPending = true;
    _feedback('Đang xử lý…');
    try {
      await _api.updateTripStatus(trip['id'] as int, status);
      _feedback(status == 'running'
          ? 'Đã bắt đầu chuyến đi.'
          : status == 'completed'
              ? 'Đã hoàn thành chuyến đi.'
              : 'Đã hủy chuyến đi.');
      try {
        await _refresh();
      } catch (_) {
        _feedback(
            'Đã lưu thay đổi, nhưng chưa tải lại được danh sách. Kéo xuống để làm mới.');
      }
    } catch (error) {
      _feedback('Không thực hiện được thao tác: $error');
    } finally {
      _actionPending = false;
    }
  }
}

class EvidenceVideoDialog extends StatefulWidget {
  const EvidenceVideoDialog({super.key, required this.file});
  final File file;

  @override
  State<EvidenceVideoDialog> createState() => _EvidenceVideoDialogState();
}

class _EvidenceVideoDialogState extends State<EvidenceVideoDialog> {
  late final VideoPlayerController _controller;
  late final Future<void> _ready;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.file(widget.file);
    _ready = _controller.initialize().then((_) {
      _controller.setLooping(false);
      _controller.play();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AlertDialog(
        title: const Text('Clip bằng chứng'),
        content: FutureBuilder<void>(
          future: _ready,
          builder: (context, snapshot) {
            if (snapshot.hasError) {
              return Text('Không giải mã được clip: ${snapshot.error}');
            }
            if (snapshot.connectionState != ConnectionState.done) {
              return const SizedBox(
                  height: 180,
                  child: Center(child: CircularProgressIndicator()));
            }
            return AspectRatio(
              aspectRatio: _controller.value.aspectRatio,
              child: Stack(alignment: Alignment.bottomCenter, children: [
                VideoPlayer(_controller),
                VideoProgressIndicator(_controller, allowScrubbing: true),
                Center(
                    child: IconButton.filledTonal(
                  icon: Icon(_controller.value.isPlaying
                      ? Icons.pause
                      : Icons.play_arrow),
                  onPressed: () => setState(() {
                    _controller.value.isPlaying
                        ? _controller.pause()
                        : _controller.play();
                  }),
                )),
              ]),
            );
          },
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Đóng'))
        ],
      );
}
