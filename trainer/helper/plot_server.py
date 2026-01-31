"""
Simple HTTP server for live plot visualization.
return
"""

import os
import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

_server_instance = None
_server_lock = threading.Lock()

PLOTS_DIR = os.path.abspath(os.path.join("saved_data", "plots"))
LIVE_PLOTS_DIR = os.path.join(PLOTS_DIR, "live_plots")
MANIFEST_PATH = os.path.join(LIVE_PLOTS_DIR, "manifest.json")
INDEX_PATH = os.path.join(LIVE_PLOTS_DIR, "index.html")
DEFAULT_PORT = 8765

CFG_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"


class PlotRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging


def _run_server(port: int):
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    handler = partial(PlotRequestHandler, directory=LIVE_PLOTS_DIR)
    server = HTTPServer(("127.0.0.1", port), handler)
    server.serve_forever()


def start_plot_server(port: int = DEFAULT_PORT, open_browser: bool = True) -> str:
    global _server_instance
    with _server_lock:
        if _server_instance is not None:
            return f"http://127.0.0.1:{port}/index.html"
        
        _ensure_manifest_exists()

        thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
        thread.start()
        _server_instance = thread

        url = f"http://127.0.0.1:{port}/index.html"
        if open_browser:
            webbrowser.open(url)
        print(f"{CFG_GREEN}[PlotServer] Started at {url}{COLOR_RESET}")
        return url


def _ensure_manifest_exists():
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH):
        return
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"plots": {}}, f)


def update_manifest(plots: dict[str, list[str]], persistent_categories: set[str] | None = None):
    """
    Update the manifest with new plots.
    plots: dict mapping category -> list of relative filepaths (from live_plots dir)
    persistent_categories: set of category names that are persistent (multi-seed)
    return
    """
    os.makedirs(LIVE_PLOTS_DIR, exist_ok=True)
    existing: dict[str, list[str]] = {}
    existing_persistent: list[str] = []
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            data = json.load(f)
            existing = data.get("plots", {})
            existing_persistent = data.get("persistent_categories", [])

    existing.update(plots)
    
    if persistent_categories:
        for cat in persistent_categories:
            if cat not in existing_persistent:
                existing_persistent.append(cat)
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump({
            "plots": existing,
            "persistent_categories": existing_persistent,
            "timestamp": _get_timestamp()
        }, f, indent=2)


def _get_timestamp() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


