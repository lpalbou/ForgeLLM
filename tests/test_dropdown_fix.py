#!/usr/bin/env python3

import asyncio
from playwright.async_api import async_playwright

async def test_dropdown_fix():
    """Test that the model dropdown is now working correctly."""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("🌐 Opening web interface...")
        await page.goto('http://localhost:5002')
        await page.wait_for_load_state('networkidle')
        
        # Click on Testing tab
        print("🎯 Switching to Testing tab...")
        await page.click('[data-bs-target="#testing"]')
        await asyncio.sleep(1)
        
        # Check if model dropdown has options
        print("🔍 Checking model dropdown...")
        option_count = await page.evaluate('''() => {
            const select = document.getElementById('test-model-select');
            return select ? select.options.length : 0;
        }''')
        
        print(f"📊 Model dropdown has {option_count} options")
        
        if option_count > 1:
            print("✅ SUCCESS: Model dropdown is populated!")
            
            # Try selecting a model
            print("🖱️ Selecting a model...")
            await page.select_option('#test-model-select', index=1)  # Select first actual model
            
            selected_model = await page.evaluate('''() => {
                const select = document.getElementById('test-model-select');
                return select.value;
            }''')
            
            print(f"📝 Selected model: {selected_model}")
            
            # Check if Load Model button is enabled
            load_btn_enabled = await page.evaluate('''() => {
                const btn = document.querySelector('#model-load-form button[type="submit"]');
                return btn && !btn.disabled;
            }''')
            
            print(f"🔘 Load Model button enabled: {load_btn_enabled}")
            
            if load_btn_enabled:
                print("✅ Model selection and Load Model button working correctly!")
            else:
                print("⚠️  Load Model button not enabled")
                
        else:
            print("❌ FAILED: Model dropdown is still empty")
        
        print("\n🏁 Test completed! You can now manually test the interface.")
        print("   The browser will stay open for 30 seconds for manual testing...")
        
        # Keep browser open for manual testing
        await asyncio.sleep(30)
        await browser.close()

if __name__ == '__main__':
    asyncio.run(test_dropdown_fix()) 