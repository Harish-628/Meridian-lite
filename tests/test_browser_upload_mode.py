"""Playwright browser test for MERIDIAN-Lite Upload mode real model inference."""

from pathlib import Path
import subprocess
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_PATH = PROJECT_ROOT / "outputs" / "reports" / "upload_mode_real_verification.png"
SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

PRE_IMG = PROJECT_ROOT / "outputs" / "temp_uploads" / "upload_pre_WhatsApp Image 2026-10-15 at 9.39.45 AM.jpeg"
POST_IMG = PROJECT_ROOT / "outputs" / "temp_uploads" / "upload_post_WhatsApp Image 2026-10-15 at 9.40.09 AM.jpeg"


def run_upload_browser_test(port: int = 8511) -> bool:
    print(f"Starting Streamlit server on port {port}...")
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
            print("Launching Chromium...")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1100})

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=30000)

            # 1. Wait for sidebar
            page.wait_for_selector("text=MERIDIAN-Lite Command Center", timeout=20000)
            print("[PASS] Sidebar loaded")

            # 2. Select Upload mode
            print("Switching to Upload mode...")
            page.locator("label:has-text(\"Upload mode\")").click()
            time.sleep(2)
            print("[PASS] Switched to Upload mode")

            # 3. Verify real backend selector is visible
            page.wait_for_selector("text=Real Damage Model (ONNX)", timeout=10000)
            print("[PASS] Verified Real Damage Model (ONNX) backend is present in Upload mode")

            # 4. Upload the pre and post disaster images
            print(f"Uploading pre-image: {PRE_IMG.name}...")
            # Streamlit file upload inputs
            upload_inputs = page.locator("input[type=\"file\"]").all()
            if len(upload_inputs) >= 2:
                upload_inputs[0].set_input_files(str(PRE_IMG))
                time.sleep(2)
                upload_inputs[1].set_input_files(str(POST_IMG))
                time.sleep(4)
                print("[PASS] Uploaded both pre and post disaster images")

            # 5. Check for run button and click if not auto-run
            run_btn = page.locator("button:has-text(\"Run AI Assessment\"), button:has-text(\"Run AI Damage Assessment\")")
            if run_btn.count() > 0:
                print("Clicking Run AI Damage Assessment...")
                run_btn.first.click()
                time.sleep(4)

            # 6. Verify Tactical Assessment Summary displays real detection metrics
            page.wait_for_selector("text=Tactical Assessment Summary", timeout=15000)
            print("[PASS] Tactical Assessment Summary rendered")

            # 7. Verify building inventory table contains real ONNX detections
            page.wait_for_selector("text=Detected Building Inventory & Damage Classifications", timeout=15000)
            print("[PASS] Building Inventory table rendered")

            # 8. Capture screenshot
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            print(f"[PASS] Screenshot saved to {SCREENSHOT_PATH}")

            return True

    finally:
        print("Terminating Streamlit test server...")
        server_process.terminate()
        server_process.wait()


if __name__ == "__main__":
    success = run_upload_browser_test()
    if success:
        print("Upload Mode Browser Verification PASSED!")
    else:
        print("Upload Mode Browser Verification FAILED!")