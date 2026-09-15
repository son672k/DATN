import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_app_module", ROOT / "scripts" / "11_run_app.py")
run_app_module = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(run_app_module)


class RunAppPortSelectionTests(unittest.TestCase):
    def test_find_free_port_skips_taken_ports(self):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            first.bind(("127.0.0.1", 0))
            base_port = first.getsockname()[1]
            second.bind(("127.0.0.1", base_port + 1))
            chosen = run_app_module.find_free_port(base_port, max_tries=3)
            self.assertNotIn(chosen, (base_port, base_port + 1))
        finally:
            first.close()
            second.close()

    def test_wait_for_url_requires_successful_http_response(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ready")

            def log_message(self, _format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/"
            self.assertTrue(run_app_module.wait_for_url(url, timeout_seconds=1))
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
