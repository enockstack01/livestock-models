"""Serve the static web app:  python web_app/serve.py [--port 8080]

Opens on http://127.0.0.1:8080, which is the origin both APIs allow by default (CORS_ORIGINS overrides it).
Plain static files only; the app talks to the Vision API and the LLM API directly from the browser.
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")            # always the current files while developing
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src *; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        super().end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    handler = functools.partial(Handler, directory=str(HERE))
    print(f"LivestockPro web app on http://{a.host}:{a.port}  (Vision API :8001, LLM API :8002 by default; change in Settings)")
    ThreadingHTTPServer((a.host, a.port), handler).serve_forever()
