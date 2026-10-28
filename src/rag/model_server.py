"""Optional project-owned Ollama process; existing external servers are never stopped.

The process registry lives in this Python process. After a crash, a surviving
server must be stopped manually before it can be managed again.
"""
import atexit
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

from .models import RagError


PROJECT_URL = "http://127.0.0.1:11435"
_lock = threading.Lock()
_servers = {}


def storage_directory(root):
    return Path(root).resolve() / "data/models"


def managed_server(root):
    entry = _servers.get(str(Path(root).resolve()))
    return entry if entry and entry["process"].poll() is None else None


def _binary():
    located = shutil.which("ollama")
    if located:
        return located
    for path in ("/Applications/Ollama.app/Contents/Resources/ollama", "/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"):
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path
    raise RagError("Ollama is not installed or its CLI was not found. Install Ollama, then try again.")


def _ready():
    with requests.Session() as session:
        session.trust_env = False
        try:
            response = session.get(PROJECT_URL + "/api/version", timeout=(0.5, 1), allow_redirects=False)
            value = response.json()
            return response.status_code == 200 and isinstance(value, dict) and bool(value.get("version"))
        except (requests.RequestException, ValueError):
            return False


def start_server(root):
    root = Path(root).resolve()
    with _lock:
        if managed_server(root):
            return PROJECT_URL
        binary = _binary()
        # Do not assume that a responding process belongs to this project.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 11435))
            except OSError as error:
                raise RagError("Port 11435 is already occupied by a server this GUI does not own. It will not be replaced. Stop that server and retry, or use an existing local endpoint.") from error
        directory = storage_directory(root)
        directory.mkdir(parents=True, exist_ok=True)
        logs = root / "data/model-server"
        logs.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(OLLAMA_MODELS=str(directory), OLLAMA_HOST="127.0.0.1:11435",
                           OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NUM_PARALLEL="1", OLLAMA_NO_CLOUD="1")
        with (logs / "ollama.log").open("ab") as output:
            process = subprocess.Popen([binary, "serve"], env=environment, cwd=root,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != "nt"))
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RagError(f"Project Ollama server exited. Inspect {logs / 'ollama.log'}.")
                if _ready() and process.poll() is None:
                    _servers[str(root)] = {"process": process, "directory": str(directory)}
                    return PROJECT_URL
                time.sleep(0.25)
            raise RagError(f"Project Ollama server did not become ready. Inspect {logs / 'ollama.log'}.")
        except BaseException:
            _terminate(process)
            raise


def _terminate(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def stop_server(root):
    with _lock:
        entry = _servers.pop(str(Path(root).resolve()), None)
        if entry:
            _terminate(entry["process"])


@atexit.register
def _cleanup():
    for root in list(_servers):
        stop_server(root)
