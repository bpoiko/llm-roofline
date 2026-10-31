"""The run watchdog against a fake Ollama that hangs until the model is unloaded,
reproducing the 9-hour hang from memory thrash."""

import http.server
import json
import threading
import time

import pytest

from ollama_bench import RunTimeout, generate


@pytest.fixture
def hanging_ollama():
    unloaded = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("keep_alive") == 0:
                unloaded.set()
                self._reply(200, {})
            elif body["prompt"] == "fast":
                self._reply(200, {"eval_count": 1})
            else:
                unloaded.wait(10)  # a real runner errors out when it is unloaded mid-request
                self._reply(500, {"error": "runner terminated"})

        def _reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", unloaded
    srv.shutdown()


def test_watchdog_unloads_and_raises(hanging_ollama):
    host, unloaded = hanging_ollama
    t0 = time.monotonic()
    with pytest.raises(RunTimeout):
        generate(host, "m", "hang", {}, timeout=0.3)
    assert unloaded.is_set()
    assert time.monotonic() - t0 < 5


def test_fast_request_is_untouched(hanging_ollama):
    host, unloaded = hanging_ollama
    resp, wall = generate(host, "m", "fast", {}, timeout=5)
    assert resp == {"eval_count": 1}
    time.sleep(0.1)
    assert not unloaded.is_set()
