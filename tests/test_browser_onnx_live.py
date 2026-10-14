"""
Playwright live browser automation test for MERIDIAN-Lite ONNX backend verification.
Executes full end-to-end browser workflow:
1. Starts Streamlit on port 8505
2. Selects 'Real Model Mode'
3. Selects 'Real Damage Model (ONNX)' and Hurricane Michael dataset
4. Clicks 'Run AI Damage Assessment'
5. Waits for completion toast
6. Verifies metrics, priority zones, map iframe, and inventory table
7. Downloads GeoJSON to outputs/temp_downloads/browser_onnx_damage.geojson
8. Validates downloaded GeoJSON contains real ONNX properties
9. Saves screenshot to outputs/reports/onnx_browser_verification.png
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_PATH = PROJECT_ROOT / "outputs" / "reports" / "onnx_browser_verification.png"
DOWNLOADS_DIR = PROJECT_ROOT / "outputs" / "temp_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_OUT = DOWNLOADS_DIR / "browser_onnx_damage.geojson"


def run_onnx_browser_verification(port: int = 8505) -> bool:
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
        # Give server time to initialize
        time.sleep(4)

        with sync_playwright() as p:
            print("Launching Chromium browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=30000)

            # 1. Wait for sidebar
            page.wait_for_selector("text=MERIDIAN-Lite Command Center", timeout=20000)
            print("[PASS] Confirmed sidebar title")

            # 2. Select 'Real Model Mode'
            print("Selecting 'Real Model Mode'...")
            real_mode_radio = page.locator("label:has-text('Real Model Mode')")
            real_mode_radio.click()
            time.sleep(1)

            # 3. Verify 'REAL MODEL INFERENCE OUTPUT' badge
            page.wait_for_selector("text=REAL MODEL INFERENCE OUTPUT", timeout=10000)
            print("[PASS] Confirmed 'REAL MODEL INFERENCE OUTPUT' badge")

            # 4. Confirm Model Backend is 'Real Damage Model (ONNX)'
            page.wait_for_selector("text=Real Damage Model (ONNX)", timeout=10000)
            print("[PASS] Confirmed 'Real Damage Model (ONNX)' backend selected")

            # 5. Confirm dataset is Hurricane Michael
            page.wait_for_selector("text=Hurricane Michael", timeout=10000)
            print("[PASS] Confirmed Hurricane Michael dataset selected")

            # 6. Click 'Run AI Damage Assessment'
            run_btn = page.locator("button:has-text('Run AI Damage Assessment')")
            assert run_btn.is_visible(), "Run AI Damage Assessment button not visible"
            print("Clicking 'Run AI Damage Assessment'...")
            run_btn.click()

            # 7. Wait for completion toast
            page.wait_for_selector("text=Real model assessment completed", timeout=30000)
            print("[PASS] Confirmed execution completion toast")

            # 8. Verify metrics
            page.wait_for_selector("text=Total Buildings Detected", timeout=10000)
            page.wait_for_selector("text=Mean Assessment Confidence", timeout=10000)
            print("[PASS] Confirmed Tactical Assessment Summary metrics")

            # 9. Verify Priority Zones section
            page.wait_for_selector("text=Stage 2: Priority-Zone Scoring & Resource Allocation", timeout=10000)
            page.wait_for_selector("text=Top Priority Zone", timeout=10000)
            print("[PASS] Confirmed Priority-Zone section and metric cards")

            # 10. Verify Folium Map iframe
            map_frame = page.locator("iframe").first
            map_frame.wait_for(state="attached", timeout=15000)
            print("[PASS] Confirmed interactive Folium map iframe")

            # 11. Verify building inventory table
            page.wait_for_selector("text=Detected Building Inventory & Damage Classifications", timeout=10000)
            print("[PASS] Confirmed building inventory table")

            # 12. Download GeoJSON
            geo_btn = page.locator("button:has-text('Download GeoJSON')")
            geo_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                geo_btn.click()
            download = download_info.value
            download.save_as(str(GEOJSON_OUT))
            assert GEOJSON_OUT.exists() and GEOJSON_OUT.stat().st_size > 0
            print(f"[PASS] Downloaded GeoJSON to {GEOJSON_OUT} ({GEOJSON_OUT.stat().st_size} bytes)")

            # 13. Verify downloaded GeoJSON content
            with open(GEOJSON_OUT, "r") as f:
                d_geo = json.load(f)
            assert d_geo.get("type") == "FeatureCollection"
            feats = d_geo.get("features", [])
            assert len(feats) > 0, "No features found in downloaded GeoJSON"
            first_props = feats[0]["properties"]
            assert first_props.get("backend_name") == "real_damage_model", f"Wrong backend_name: {first_props.get('backend_name')}"
            assert first_props.get("model_type") == "lightweight_bitemporal_onnx", f"Wrong model_type: {first_props.get('model_type')}"
            assert first_props.get("is_simulated") is False, f"Expected is_simulated=False"
            print(f"[PASS] Downloaded GeoJSON verified: {len(feats)} features from ONNX backend")

            # 14. Save screenshot
            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            assert SCREENSHOT_PATH.exists() and SCREENSHOT_PATH.stat().st_size > 10000
            print(f"[PASS] Captured screenshot to {SCREENSHOT_PATH} ({SCREENSHOT_PATH.stat().st_size} bytes)")

            browser.close()
            return True

    finally:
        print("Stopping Streamlit server...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    success = run_onnx_browser_verification(port=8505)
    sys.exit(0 if success else 1)
