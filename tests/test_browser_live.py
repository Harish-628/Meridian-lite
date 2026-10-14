"""
Playwright live browser automation test for MERIDIAN-Lite Streamlit Dashboard.
Verifies:
1. Page title & sidebar title 'MERIDIAN-Lite Command Center'
2. Demo mode selection & warning banner
3. 'Run AI Damage Assessment' button click
4. Metrics cards rendering (Total Buildings, Severe %, Mean Confidence)
5. Imagery comparison & visual difference overlay rendering
6. Folium interactive GIS map rendering
7. Detected building inventory table rendering
8. Live downloads for GeoJSON, Summary JSON, CSV, and Overlay
9. Captures dashboard screenshot for visual release verification
"""

import os
import subprocess
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_PATH = PROJECT_ROOT / "outputs" / "reports" / "browser_dashboard_screenshot.png"
DOWNLOADS_DIR = PROJECT_ROOT / "outputs" / "temp_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def run_browser_test():
    port = 8503
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

    print(f"Starting Streamlit server on port {port}...")
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
            print("Launching headless Chromium browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()

            url = f"http://localhost:{port}"
            print(f"Navigating to {url}...")
            page.goto(url, timeout=30000)

            # Wait for main page element or sidebar to appear
            page.wait_for_selector("text=MERIDIAN-Lite Command Center", timeout=20000)
            print("[PASS] Confirmed sidebar title: 'MERIDIAN-Lite Command Center'")

            # Verify main header
            page.wait_for_selector("text=MERIDIAN-Lite", timeout=10000)
            print("[PASS] Confirmed main header: 'MERIDIAN-Lite'")

            # Verify Demo mode warning
            demo_warn = page.locator("text=Demo mode uses simulated imagery").first
            assert demo_warn.is_visible()
            print("[PASS] Confirmed Demo Mode warning banner")

            # Click "Run AI Damage Assessment" button
            page.wait_for_selector("button:has-text('Run AI Damage Assessment')", timeout=15000)
            run_btn = page.locator("button:has-text('Run AI Damage Assessment')")
            assert run_btn.is_visible()
            print("Clicking 'Run AI Damage Assessment'...")
            run_btn.click()

            # Wait for execution toast or metrics update
            page.wait_for_selector("text=Total Buildings Detected", timeout=20000)
            print("[PASS] Confirmed Metrics card: 'Total Buildings Detected'")

            # Verify metrics values
            page.wait_for_selector("text=Destroyed / Major Damage", timeout=10000)
            page.wait_for_selector("text=Mean Assessment Confidence", timeout=10000)
            print("[PASS] Confirmed severe damage and confidence metric cards")

            # Verify visual difference caption
            page.wait_for_selector("text=Visual Comparison (Pixel Difference Overlay)", timeout=10000)
            print("[PASS] Confirmed Visual Comparison overlay section")

            # Verify interactive Folium map container
            print("Waiting for Folium map iframe...")
            map_frame = page.locator("iframe").first
            map_frame.wait_for(state="attached", timeout=15000)
            print("[PASS] Confirmed interactive Folium GIS map iframe")

            # Verify building inventory table
            page.wait_for_selector("text=Detected Building Inventory", timeout=10000)
            print("[PASS] Confirmed Detected Building Inventory section")

            # Test Downloads
            # 1. Download GeoJSON
            geo_btn = page.locator("button:has-text('Download GeoJSON')")
            geo_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                geo_btn.click()
            download_geo = download_info.value
            geo_save_path = DOWNLOADS_DIR / "browser_test_damage.geojson"
            download_geo.save_as(str(geo_save_path))
            assert geo_save_path.exists() and geo_save_path.stat().st_size > 0
            print(f"[PASS] Verified GeoJSON download ({geo_save_path.stat().st_size} bytes)")

            # 2. Download Summary JSON
            json_btn = page.locator("button:has-text('Download Summary JSON')")
            json_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                json_btn.click()
            download_json = download_info.value
            json_save_path = DOWNLOADS_DIR / "browser_test_summary.json"
            download_json.save_as(str(json_save_path))
            assert json_save_path.exists() and json_save_path.stat().st_size > 0
            print(f"[PASS] Verified Summary JSON download ({json_save_path.stat().st_size} bytes)")

            # 3. Download CSV
            csv_btn = page.locator("button:has-text('Download CSV Results')")
            csv_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                csv_btn.click()
            download_csv = download_info.value
            csv_save_path = DOWNLOADS_DIR / "browser_test_buildings.csv"
            download_csv.save_as(str(csv_save_path))
            assert csv_save_path.exists() and csv_save_path.stat().st_size > 0
            print(f"[PASS] Verified CSV download ({csv_save_path.stat().st_size} bytes)")

            # 4. Download Damage Overlay
            overlay_btn = page.locator("button:has-text('Download Damage Overlay')")
            overlay_btn.scroll_into_view_if_needed()
            with page.expect_download() as download_info:
                overlay_btn.click()
            download_overlay = download_info.value
            overlay_save_path = DOWNLOADS_DIR / "browser_test_overlay.png"
            download_overlay.save_as(str(overlay_save_path))
            assert overlay_save_path.exists() and overlay_save_path.stat().st_size > 0
            print(f"[PASS] Verified Overlay PNG download ({overlay_save_path.stat().st_size} bytes)")

            # Capture visual screenshot
            page.screenshot(path=str(SCREENSHOT_PATH), full_page=True)
            print(f"[PASS] Saved full page dashboard screenshot to: {SCREENSHOT_PATH}")

            browser.close()
            print("=" * 60)
            print("LIVE BROWSER AUTOMATION TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)

    finally:
        print("Terminating Streamlit server process...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    run_browser_test()
