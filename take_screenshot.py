from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto("http://localhost:8501")
        
        print("Waiting for Streamlit app to load...")
        page.wait_for_selector("text=MediScan AI")
        time.sleep(3)
        
        # Click "Load Sample" if it exists
        try:
            print("Clicking Load Sample...")
            page.locator("button", has_text="Load Sample").click(timeout=3000)
            time.sleep(1)
        except Exception as e:
            print("Load Sample not found or failed:", e)

        # Click the Tabs
        try:
            print("Clicking Step 3 tab...")
            page.locator("button", has_text="Results").click(timeout=3000)
            time.sleep(1)
        except:
            pass

        try:
            print("Clicking Generate Diagnosis...")
            page.locator("button", has_text="Generate Diagnosis").click(timeout=5000)
            time.sleep(1)
            # Wait for spinner to disappear
            page.wait_for_selector("text=Analyzing... Please wait", state="hidden", timeout=30000)
            time.sleep(2)
        except Exception as e:
            print("Generate Diagnosis not found or failed:", e)

        print("Taking screenshot...")
        page.screenshot(path="demo_screenshot.png", full_page=True)
        print("Screenshot saved to demo_screenshot.png")
        browser.close()

if __name__ == "__main__":
    run()
