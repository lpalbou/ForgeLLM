#!/usr/bin/env python3

import asyncio
import time
from playwright.async_api import async_playwright

async def test_monitoring_polling():
    """Test that monitoring tab makes exactly 1 request when switching and then polls every 10s"""
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Track API calls
        api_calls = []
        
        async def track_api_calls(route):
            if '/api/dashboard/realtime' in route.request.url:
                timestamp = time.time()
                api_calls.append({
                    'timestamp': timestamp,
                    'url': route.request.url,
                    'method': route.request.method
                })
                print(f"[{time.strftime('%H:%M:%S')}] API Call: {route.request.method} {route.request.url}")
            # Let the request continue
            await route.continue_()
        
        # Intercept API calls
        await page.route('**', track_api_calls)
        
        # Navigate to the app
        print("Opening ForgeLL interface...")
        await page.goto('http://localhost:5002')
        
        # Wait for page to load
        await page.wait_for_selector('#training-tab')
        print("Page loaded, currently on Training tab")
        
        # Clear API calls from initial page load
        api_calls.clear()
        print("Cleared initial API calls, starting monitoring...")
        
        # Switch to monitoring tab
        print("\n=== SWITCHING TO MONITORING TAB ===")
        await page.click('#monitoring-tab')
        
        # Wait a moment for tab switch
        await asyncio.sleep(2)
        
        # Count calls immediately after tab switch
        immediate_calls = len(api_calls)
        print(f"Immediate calls after tab switch: {immediate_calls}")
        
        if immediate_calls == 2:
            print("❌ ISSUE CONFIRMED: 2 calls instead of 1 when switching to monitoring tab")
        elif immediate_calls == 1:
            print("✅ Good: 1 call when switching to monitoring tab")
        else:
            print(f"⚠️  Unexpected: {immediate_calls} calls when switching to monitoring tab")
        
        # Wait 15 seconds and check for periodic polling
        print("\n=== WAITING 15 SECONDS FOR PERIODIC POLLING ===")
        await asyncio.sleep(15)
        
        # Count total calls after waiting
        total_calls = len(api_calls)
        polling_calls = total_calls - immediate_calls
        
        print(f"Total calls after 15s: {total_calls}")
        print(f"Polling calls (should be 1-2): {polling_calls}")
        
        if polling_calls == 0:
            print("❌ ISSUE CONFIRMED: No periodic polling after tab switch")
        elif polling_calls >= 1:
            print("✅ Good: Periodic polling is working")
        
        # Print all API calls with timestamps
        print("\n=== ALL API CALLS ===")
        for i, call in enumerate(api_calls):
            print(f"{i+1}. [{time.strftime('%H:%M:%S', time.localtime(call['timestamp']))}] {call['method']} {call['url']}")
        
        # Calculate time gaps between calls
        if len(api_calls) > 1:
            print("\n=== TIME GAPS BETWEEN CALLS ===")
            for i in range(1, len(api_calls)):
                gap = api_calls[i]['timestamp'] - api_calls[i-1]['timestamp']
                print(f"Gap {i}: {gap:.2f} seconds")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_monitoring_polling()) 