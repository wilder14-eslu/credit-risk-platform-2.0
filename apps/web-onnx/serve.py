"""Servidor local ultrarrápido para Credit Risk ONNX Edge Web App.

Inicia un servidor HTTP optimizado con soporte para WebAssembly y abre el navegador automáticamente.
Uso:
    python serve.py
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


class WASMHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def end_headers(self) -> None:
        # Headers para habilitar WebAssembly y SharedArrayBuffer si se requiere
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".onnx"):
            return "application/octet-stream"
        if path.endswith(".json"):
            return "application/json"
        return super().guess_type(path)


def find_free_port(start_port: int = 8080) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start_port


def main() -> None:
    port = find_free_port(8502)
    url = f"http://127.0.0.1:{port}"
    print("=" * 65)
    print(" 🚀 Credit Risk Platform 2.0 · ONNX Edge Web Application")
    print(f" 🌐 URL Local: {url}")
    print(f" 📁 Directorio: {APP_DIR}")
    print(" ⚡ Carga inmediata: sin cold-start ni backend de Python.")
    print(" 💻 Presiona Ctrl+C para detener el servidor.")
    print("=" * 65)

    webbrowser.open(url)

    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    with ThreadingServer(("127.0.0.1", port), WASMHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido.")


if __name__ == "__main__":
    main()
