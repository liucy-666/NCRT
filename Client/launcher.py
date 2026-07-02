"""
NCRT v3 Desktop — 桌面应用启动器

双击此脚本或运行: python launcher.py
自动启动后端服务，打开原生桌面窗口。
"""

import sys
import os
import time
import threading
import requests

# 确保能 import 项目代码
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from Client.server import app


def main():
    # ── 1. 启动 Flask ──
    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    # ── 2. 等 Flask 就绪 ──
    print("  Waiting for server...", end="", flush=True)
    for _ in range(15):
        try:
            if requests.get("http://127.0.0.1:5000/api/health", timeout=1).status_code == 200:
                print(" ready!")
                break
        except: pass
        time.sleep(1)
        print(".", end="", flush=True)

    # ── 3. 打开窗口 ──
    import webview
    window = webview.create_window(
        title="NCRT v3 — LLM Red-Teaming Platform",
        url="http://127.0.0.1:5000",
        width=1280,
        height=800,
        min_size=(900, 600),
        text_select=True,
        confirm_close=True,
    )
    webview.start(gui="edgechromium")  # Windows: 使用系统 Edge/Chromium 内核
    # pywebview 窗口关闭后，程序自动退出


if __name__ == "__main__":
    main()
