import 'package:dms_mobile/services/dms_api_client.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('manager API error keeps its Vietnamese message', () {
    final error = DmsApiException('Chỉ dành cho supervisor');
    expect(error.toString(), 'Chỉ dành cho supervisor');
  });
}
