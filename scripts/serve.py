"""Serve the StepInside gallery locally and open it in the browser.

    python scripts/serve.py            # http://localhost:8000/site/
    python scripts/serve.py --port 9000 --no-open

Rebuilds site/examples.json first, so newly built scenes show up.
"""

import argparse
import functools
import http.server
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_site.py")], check=True)
    port = free_port(a.port)
    url = f"http://localhost:{port}/site/"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *args: None  # quiet
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"StepInside gallery at {url}  (Ctrl+C to stop)")
        if not a.no_open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
