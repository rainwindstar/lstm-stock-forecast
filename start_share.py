# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_DIR = Path(__file__).resolve().parent
APP_PATH = APP_DIR / "app.py"
VENV_PY = APP_DIR / "venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY if VENV_PY.exists() else sys.executable)
PORT = int(os.getenv("STREAMLIT_PORT", "8502"))
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN", "").strip() or os.getenv("NGROK_TOKEN", "").strip()


def stop_existing_port_process(port):
    try:
        import psutil
    except Exception:
        return

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections(kind="inet"):
                if conn.laddr and conn.laddr.port == port:
                    proc.kill()
        except Exception:
            continue


def start_streamlit():
    print("[1/3] Streamlit 앱 시작 중...")
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    proc = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
        ],
        creationflags=flags,
    )
    print(f"      PID: {proc.pid}")
    print(f"      로컬 주소: http://localhost:{PORT}")
    return proc


def main():
    stop_existing_port_process(PORT)
    proc = start_streamlit()
    time.sleep(5)

    if not NGROK_TOKEN:
        print("[2/3] ngrok 토큰이 없어 공유 터널은 열지 않습니다.")
        print("      환경변수 NGROK_AUTHTOKEN 또는 NGROK_TOKEN을 설정하면 외부 공유가 가능합니다.")
        print("[3/3] 로컬 실행 완료")
        print("\n종료: Ctrl+C")
        try:
            while True:
                time.sleep(60)
                print(f"  실행 중... http://localhost:{PORT}")
        except KeyboardInterrupt:
            proc.terminate()
            print("종료됨.")
        return

    print("[2/3] ngrok 터널 연결 중...")
    try:
        from pyngrok import conf, ngrok
    except ModuleNotFoundError:
        proc.terminate()
        print("pyngrok가 설치되어 있지 않습니다. requirements.txt 설치 후 다시 실행해 주세요.")
        return

    conf.get_default().auth_token = NGROK_TOKEN
    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url

    print("[3/3] 완료!")
    print()
    print("=" * 60)
    print("  아래 주소를 공유하세요.")
    print()
    print(f"  >>  {public_url}")
    print()
    print("  주의: 이 창을 닫으면 접속이 종료됩니다.")
    print("=" * 60)

    try:
        subprocess.run(["clip"], input=public_url.encode("ascii"), check=True)
        print("  (URL이 클립보드에 복사됨)")
    except Exception:
        pass

    print("\n종료: Ctrl+C")
    try:
        while True:
            time.sleep(60)
            print(f"  공유 중... {public_url}")
    except KeyboardInterrupt:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        proc.terminate()
        print("종료됨.")


if __name__ == "__main__":
    main()
