# -*- coding: utf-8 -*-
"""Run Streamlit and optionally share it through ngrok or Cloudflare."""
import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlopen

APP_DIR = Path(__file__).resolve().parent
APP_PATH = APP_DIR / "app.py"
VENV_PY = APP_DIR / "venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY if VENV_PY.exists() else sys.executable)
PORT = int(os.getenv("STREAMLIT_PORT", "8502"))
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN", "").strip() or os.getenv("NGROK_TOKEN", "").strip()


def start_streamlit():
    return subprocess.Popen([
        PYTHON, "-m", "streamlit", "run", str(APP_PATH),
        "--server.address", "127.0.0.1", "--server.port", str(PORT),
        "--server.headless", "true",
    ], cwd=APP_DIR)


def wait_ready(proc, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Streamlit exited before becoming ready")
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("Streamlit readiness timed out")


def cloudflare_url(proc, timeout=60):
    lines = queue.Queue()

    def read_output():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=min(1, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            continue
        if line is None:
            raise RuntimeError("cloudflared exited without a public URL")
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com\b", line)
        if match and match.group(0) != "https://api.trycloudflare.com":
            return match.group(0)
    raise RuntimeError("Cloudflare URL creation timed out")


def save_url(public_url):
    (APP_DIR / "public_url.txt").write_text(public_url + "\n", encoding="utf-8")
    print(f"공유 주소: {public_url}")
    if os.name == "nt":
        try:
            subprocess.run(["clip.exe"], input=public_url.encode("ascii"), check=True)
            print("URL이 클립보드에 복사됨")
        except (OSError, subprocess.CalledProcessError):
            print("클립보드 복사 실패: 위 주소 또는 public_url.txt를 복사하세요.")


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ngrok", "cloudflare"], default="ngrok")
    args = parser.parse_args(argv)
    proc = tunnel_proc = ngrok = tunnel = None
    try:
        # Never leave a previous session's URL looking like a current one.
        (APP_DIR / "public_url.txt").unlink(missing_ok=True)
        if args.provider == "cloudflare":
            executable = shutil.which("cloudflared")
            if not executable:
                raise RuntimeError("cloudflared가 필요합니다: winget install --id Cloudflare.cloudflared -e")
        elif NGROK_TOKEN:
            try:
                from pyngrok import conf, ngrok
            except ModuleNotFoundError as exc:
                raise RuntimeError("python -m pip install -r requirements.txt 실행 후 다시 시도하세요.") from exc
            conf.get_default().auth_token = NGROK_TOKEN
        # Do not reuse or kill another application's listener.
        import socket
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", PORT))
        proc = start_streamlit()
        wait_ready(proc)
        print(f"로컬 주소: http://localhost:{PORT}")
        if args.provider == "cloudflare":
            tunnel_proc = subprocess.Popen([
                executable, "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                "--no-autoupdate",
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", cwd=APP_DIR)
            save_url(cloudflare_url(tunnel_proc))
        elif ngrok is not None:
            tunnel = ngrok.connect(PORT, "http", bind_tls=True)
            save_url(tunnel.public_url)
        else:
            print("ngrok 토큰 없음: 로컬 실행만 합니다. NGROK_AUTHTOKEN을 설정하면 공유할 수 있습니다.")
        print("종료: Ctrl+C. 이 창을 닫으면 공유가 종료됩니다.")
        while True:
            if proc.poll() is not None or (tunnel_proc is not None and tunnel_proc.poll() is not None):
                raise RuntimeError("Streamlit or Cloudflare process exited")
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"공유 실행 실패: {exc}", file=sys.stderr)
        return 1
    finally:
        stop(tunnel_proc)
        try:
            if ngrok is not None:
                if tunnel is not None:
                    ngrok.disconnect(tunnel.public_url)
                ngrok.kill()
        finally:
            stop(proc)
            (APP_DIR / "public_url.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
