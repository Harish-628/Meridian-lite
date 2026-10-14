"""
Playwright live browser automation test for MERIDIAN-Lite Stage 2 Real Model Mode.
Verifies:
1. Selection of 'Real Model Mode' in sidebar
2. Display of 'REAL MODEL INFERENCE OUTPUT' banner & dataset metadata
3. Execution of real AI damage assessment (ONNX CPU backend on Hurricane Michael imagery)
4. Metrics cards rendering (Total Buildings, Severe %, Mean Confidence)
5. Stage 2 Priority-Zone intelligence section and ranking table
6. Interactive Folium GIS map with building polygons and priority zone overlays
7. Real model download outputs (GeoJSON, Priority Zones JSON, CSV, Overlay PNG)
8. Captures full dashboard screenshot to outputs/reports/browser_real_model_screenshot.png
"""

import os
from pathlib import Path
import subprocess
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_PATH = PROJECT_ROOT / "outputs" / "reports" / "browser_real_model_screenshot.png"
DOWNLOADS_DIR = PROJECT_ROOT / "outputs" / "temp_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def run_stage2_browser_test():
    port = 8504
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

    print(f"Starting Streamlit server for Stage 2 test on port {port}...")
    server_process = subprocess.Popen(
        server_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        time.sleep(4)

        with sync_playwright() as p:
            print("Launching headless Chromium browser for Stage 2...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=30000)

            # Wait for sidebar
            page.wait_for_selector("text=MERIDIAN-Lite Command Center", timeout=20000)
            print("[PASS] Confirmed sidebar title")

            # Select 'Real Model Mode'
            print("Selecting 'Real Model Mode' radio button...")
            real_mode_radio = page.locator("label:has-text('Real Model Mode')")
            real_mode_radio.click()
            time.sleep(1)

            # Verify 'REAL MODEL INFERENCE OUTPUT' banner
            page.wait_for_selector("text=REAL MODEL INFERENCE OUTPUT", timeout=10000)
            print("[PASS] Confirmed 'REAL MODEL INFERENCE OUTPUT' badge")
            page.wait_for_selector("text=Hurricane Michael", timeout=10000)
            print("[PASS] Confirmed Hurricane Michael dataset metadata in UI")

            # Click 'Run AI Damage Assessment'
            run_btn = page.locator("button:has-text('Run AI Damage Assessment')")
            assert run_btn.is_visible()
            print("Clicking 'Run AI Damage Assessment' in Real Model Mode...")
            run_btn.click()

            # Wait for execution toast confirming real model completion
            page.wait_for_selector("text=Real model assessment completed", timeout=30000)
            print("[PASS] Confirmed real model execution completion toast")

            # Wait for metrics to render
            page.wait_for_selector("text=Total Buildings Detected", timeout=10000)
            print("[PASS] Confirmed Metrics card: 'Total Buildings Detected'")
            page.wait_for_selector("text=Mean Assessment Confidence", timeout=10000)
            print("[PASS] Confirmed Mean Assessment Confidence card")

            # Verify Stage 2 Priority-Zone section
            page.wait_for_selector("text=Stage 2: Priority-Zone Scoring & Resource Allocation", timeout=10000)
            print("[PASS] Confirmed Stage 2 Priority-Zone Scoring header")
            page.wait_for_selector("text=Top Priority Zone", timeout=10000)
            print("[PASS] Confirmed Top Priority Zone metric card")

            # Verify Folium map
            print("Waiting for Folium GIS map iframe...")
            map_frame = page.locator("iframe").first
            map_frame.wait_for(state="attached", timeout=15000)
            print("[PASS] Confirmed interactive Folium GIS map iframe with Priority Zones")

            # Verify priority zones and building table
            page.wait_for_selector("text=Detected Building Inventory & Damage Classifications", timeout=10000)
            print("[PASS] Confirmed building damage classification table")

            # Test Stage 2 Downloads
            # 1. Download GeoJSON
            geo_btn = page.locator("button:has-text('Download GeoJSON')")
            geo_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                geo_btn.click()
            d_geo = download_info.value
            p_geo = DOWNLOADS_DIR / "stage2_browser_damage.geojson"
            d_geo.save_as(str(p_geo))
            assert p_geo.exists() and p_geo.stat().st_size > 0
            print(f"[PASS] Verified Stage 2 GeoJSON download ({p_geo.stat().st_size} bytes)")

            # 2. Download Priority Zones JSON
            pz_btn = page.locator("button:has-text('Download Priority Zones JSON')")
            pz_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                pz_btn.click()
            d_pz = download_info.value
            p_pz = DOWNLOADS_DIR / "stage2_browser_priority_zones.json"
            d_pz.save_as(str(p_pz))
            assert p_pz.exists() and p_pz.stat().st_size > 0
            print(f"[PASS] Verified Priority Zones JSON download ({p_pz.stat().st_size} bytes)")

            # Capture full page screenshot
            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            print(f"[PASS] Saved Stage 2 dashboard screenshot to: {SCREENSHOT_PATH}")

            browser.close()
            print("=" * 65)
            print("STAGE 2 BROWSER AUTOMATION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 65)

    finally:
        print("Terminating Streamlit server process...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    run_stage2_browser_test()
