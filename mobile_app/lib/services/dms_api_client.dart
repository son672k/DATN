import 'dart:convert';
import 'package:http/http.dart' as http;

class DmsApiException implements Exception {
  DmsApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

class DmsApiClient {
  DmsApiClient(this.baseUrl, {this.token});
  final String baseUrl;
  final String? token;

  Uri _uri(String path) =>
      Uri.parse('${baseUrl.replaceAll(RegExp(r'/$'), '')}$path');
  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      };

  Map<String, dynamic> _decode(http.Response response) {
    final body = response.body.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw DmsApiException(
          body['detail']?.toString() ?? 'Lỗi HTTP ${response.statusCode}');
    }
    return body;
  }

  Future<Map<String, dynamic>> login(String username, String password) async {
    final response = await http.post(_uri('/api/auth/token'),
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: {'username': username, 'password': password});
    final body = _decode(response);
    return body;
  }

  Future<Map<String, dynamic>> me() async => Map<String, dynamic>.from(
      _decode(await http.get(_uri('/api/auth/me'), headers: _headers))['user']
          as Map);

  Future<List<Map<String, dynamic>>> _list(String path, String key) async {
    final body = _decode(await http.get(_uri(path), headers: _headers));
    return List<Map<String, dynamic>>.from(body[key] as List);
  }

  Future<List<Map<String, dynamic>>> fleet() =>
      _list('/api/admin/fleet/overview', 'drivers');
  Future<List<Map<String, dynamic>>> alerts() =>
      _list('/api/admin/alerts?limit=50', 'alerts');
  Future<List<Map<String, dynamic>>> vehicles() =>
      _list('/api/admin/vehicles', 'vehicles');
  Future<List<Map<String, dynamic>>> trips() =>
      _list('/api/admin/trips', 'trips');
  Future<List<Map<String, dynamic>>> driverAlerts() =>
      _list('/api/driver/me/alerts?limit=100', 'alerts');
  Future<List<Map<String, dynamic>>> driverTrips() =>
      _list('/api/driver/me/trips', 'trips');
  Future<Map<String, dynamic>> driverDashboard() async =>
      Map<String, dynamic>.from(_decode(
          await http.get(_uri('/api/driver/me/dashboard'), headers: _headers)));
  Future<List<Map<String, dynamic>>> tripLocations(int tripId,
          {required bool driver}) =>
      _list(
          driver
              ? '/api/driver/me/trips/$tripId/locations?limit=500'
              : '/api/admin/trips/$tripId/locations?limit=500',
          'locations');

  Future<Map<String, dynamic>> updateTripStatus(
      int tripId, String status) async {
    final response = await http.patch(_uri('/api/admin/trips/$tripId/status'),
        headers: _headers, body: jsonEncode({'status': status}));
    return Map<String, dynamic>.from(_decode(response)['trip'] as Map);
  }

  Future<Map<String, dynamic>> reviewAlert(
      int eventId, String status, String? note) async {
    final response = await http.patch(_uri('/api/events/$eventId/review'),
        headers: _headers, body: jsonEncode({'status': status, 'note': note}));
    return Map<String, dynamic>.from(_decode(response)['event'] as Map);
  }

  Future<List<int>> _evidence(String path) async {
    final response = await http.get(_uri(path), headers: _headers);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      _decode(response);
    }
    return response.bodyBytes;
  }

  Future<List<int>> snapshot(int eventId) =>
      _evidence('/api/events/$eventId/snapshot.jpg');

  Future<List<int>> clip(int eventId) =>
      _evidence('/api/events/$eventId/clip.mp4');
}
