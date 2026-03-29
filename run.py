#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
PORT_SEARCH_LIMIT = 25


def find_free_port(start_port: int) -> int:
    for port in range(start_port, start_port + PORT_SEARCH_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((HOST, port)) != 0:
                return port
    raise RuntimeError(f"Unable to find a free port starting at {start_port}.")


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        process.terminate()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def wait_for_early_exit(process: subprocess.Popen[bytes], name: str, timeout_seconds: float = 2.0) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {process.returncode}.")
        time.sleep(0.1)


def main() -> int:
    root = Path(__file__).resolve().parent
    frontend_dir = root / "frontend"
    local_python = root / ".venv" / "bin" / "python"
    backend_python = str(local_python if local_python.exists() else Path(sys.executable))

    if not frontend_dir.exists():
        print("Missing frontend directory. Expected 'frontend/' under graph-agent.", file=sys.stderr)
        return 1

    if shutil.which("npm") is None:
        print("npm is not installed or not on PATH.", file=sys.stderr)
        return 1

    if not local_python.exists():
        print(
            "Missing local virtualenv at '.venv'. Run:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/pip install -e .",
            file=sys.stderr,
        )
        return 1

    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        install_result = subprocess.run(["npm", "install"], cwd=frontend_dir)
        if install_result.returncode != 0:
            return install_result.returncode

    backend_port = find_free_port(DEFAULT_BACKEND_PORT)
    frontend_port = find_free_port(DEFAULT_FRONTEND_PORT)
    api_base_url = f"http://{HOST}:{backend_port}"
    frontend_url = f"http://{HOST}:{frontend_port}"

    backend_env = os.environ.copy()
    existing_python_path = backend_env.get("PYTHONPATH", "")
    source_path = str(root / "src")
    backend_env["PYTHONPATH"] = source_path if not existing_python_path else f"{source_path}{os.pathsep}{existing_python_path}"
    backend_env["GRAPH_AGENT_HOST"] = HOST
    backend_env["GRAPH_AGENT_PORT"] = str(backend_port)

    frontend_env = os.environ.copy()
    frontend_env["VITE_API_BASE_URL"] = api_base_url

    backend_command = [
        backend_python,
        "-m",
        "uvicorn",
        "graph_agent.api.app:app",
        "--reload",
        "--host",
        HOST,
        "--port",
        str(backend_port),
    ]
    frontend_command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        HOST,
        "--port",
        str(frontend_port),
    ]

    print(f"Starting backend: {api_base_url}")
    print(f"Starting frontend: {frontend_url}")
    print("Press Ctrl+C once to stop both processes.")

    popen_kwargs: dict[str, object] = {"start_new_session": os.name != "nt"}
    backend_process = subprocess.Popen(backend_command, cwd=root, env=backend_env, **popen_kwargs)
    frontend_process = subprocess.Popen(frontend_command, cwd=frontend_dir, env=frontend_env, **popen_kwargs)

    try:
        wait_for_early_exit(backend_process, "Backend")
        wait_for_early_exit(frontend_process, "Frontend")

        while True:
            if backend_process.poll() is not None:
                raise RuntimeError(f"Backend exited with code {backend_process.returncode}.")
            if frontend_process.poll() is not None:
                raise RuntimeError(f"Frontend exited with code {frontend_process.returncode}.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping graph-agent workload...")
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        terminate_process(frontend_process)
        terminate_process(backend_process)

        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()

        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
