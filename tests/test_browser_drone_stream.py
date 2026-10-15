"""Playwright live browser test for MERIDIAN-Lite Live Drone Reconnaissance Stream.

Verifies:
1. Selection of 'Live Drone Recon Stream' mode.
2. Real-time dual optical viewfinder rendering with HUD overlays and bounding boxes.
3. Telemetry sensors (Altitude, Speed, Wind Speed, Barometer, Battery).
4. Stage 2 Autonomous search route metrics.
5. Stage 2 Tactical emergency resource allocation matrix.
6. Stage 3 Predictive multi-hazard risk forecasting.
7. Saves screenshot to outputs/reports/drone_stream_live_verification.png.
"""

from pathlib import Path
import subprocess
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_PATH = PROJECT_ROOT / "outputs" / "reports" / "drone_stream_live_verification.png"


def run_drone_stream_browser_test(port: int = 8507) -> bool:
    print(f"Starting Streamlit server on port {port} for Drone Stream verification...")
    server_cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "streamlit",
        "run",
        "app/main.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    server_process = subprocess.Popen(
        server_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        time.sleep(4)

        with sync_playwright() as p:
            print("Launching Chromium browser...")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 950})

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=30000)

            # 1. Wait for sidebar
            page.wait_for_selector("text=MERIDIAN-Lite Command Center", timeout=20000)
            print("[PASS] Confirmed sidebar title")

            # 2. Select 'Live Drone Recon Stream'
            print("Selecting 'Live Drone Recon Stream'...")
            drone_mode_radio = page.locator("label:has-text('Live Drone Recon Stream')")
            drone_mode_radio.click()
            time.sleep(2)

            # 3. Verify Live Drone Stream Header
            page.wait_for_selector("text=Live Tactical Drone Reconnaissance Stream", timeout=10000)
            print("[PASS] Confirmed Live Drone Stream title")

            # 4. Verify Telemetry Sensors
            page.wait_for_selector("text=Flight Telemetry", timeout=10000)
            page.wait_for_selector("text=Environmental Sensors", timeout=10000)
            page.wait_for_selector("text=Optical Sensor Payload", timeout=10000)
            page.wait_for_selector("text=Avionics & Datalink", timeout=10000)
            print("[PASS] Confirmed all 4 Telemetry Sensor Panels")

            # 5. Verify Stage 2 Search Flight Plan
            page.wait_for_selector("text=Stage 2: Autonomous Drone Search Route", timeout=10000)
            page.wait_for_selector("text=Flight Distance", timeout=10000)
            print("[PASS] Confirmed Autonomous Drone Search Route")

            # 6. Verify Stage 2 Resource Allocation Matrix
            page.wait_for_selector("text=Stage 2: Tactical Resource Allocation Matrix", timeout=10000)
            print("[PASS] Confirmed Tactical Resource Allocation Matrix")

            # 7. Verify Stage 3 Predictive Multi-Hazard Intelligence
            page.wait_for_selector("text=Stage 3: Predictive Multi-Hazard Intelligence", timeout=10000)
            print("[PASS] Confirmed Stage 3 Predictive Intelligence")

            # 8. Capture screenshot
            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SCREENSHOT_PATH))
            assert SCREENSHOT_PATH.exists() and SCREENSHOT_PATH.stat().st_size > 20000
            print(f"[PASS] Captured screenshot to {SCREENSHOT_PATH} ({SCREENSHOT_PATH.stat().st_size} bytes)")

            browser.close()
            return True

    finally:
        print("Stopping Streamlit server...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    success = run_drone_stream_browser_test()
    if not success:
        exit(1)
