#!/usr/bin/env python3

import asyncio
import time
from playwright.async_api import async_playwright

async def test_dropdown_debug():
    """Debug the model dropdown issue in the web interface."""
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False, devtools=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Enable console logging
        page.on('console', lambda msg: print(f"CONSOLE: {msg.type}: {msg.text}"))
        page.on('pageerror', lambda error: print(f"PAGE ERROR: {error}"))
        
        print("🌐 Opening web interface...")
        await page.goto('http://localhost:5002')
        
        # Wait for page to load
        await page.wait_for_load_state('networkidle')
        
        print("🔍 Checking DOM elements...")
        
        # Check if test-model-select exists
        test_select = await page.query_selector('#test-model-select')
        if test_select:
            print("✅ test-model-select element found")
            
            # Get current options
            options = await page.evaluate('''() => {
                const select = document.getElementById('test-model-select');
                if (!select) return null;
                return Array.from(select.options).map(opt => ({
                    value: opt.value,
                    text: opt.textContent
                }));
            }''')
            
            print(f"📋 Current options: {len(options) if options else 0}")
            if options:
                for i, opt in enumerate(options[:5]):  # Show first 5
                    print(f"  {i}: {opt['value']} - {opt['text']}")
                if len(options) > 5:
                    print(f"  ... and {len(options) - 5} more")
            else:
                print("❌ No options found")
        else:
            print("❌ test-model-select element not found")
        
        # Check if TrainingInterface is loaded
        interface_loaded = await page.evaluate('''() => {
            return typeof window.trainingInterface !== 'undefined';
        }''')
        print(f"🔧 TrainingInterface loaded: {interface_loaded}")
        
        # Check API response
        print("🔍 Testing API call...")
        api_response = await page.evaluate('''async () => {
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                return {
                    success: true,
                    modelCount: data.models ? data.models.length : 0,
                    firstModel: data.models && data.models.length > 0 ? data.models[0] : null
                };
            } catch (error) {
                return {
                    success: false,
                    error: error.message
                };
            }
        }''')
        
        print(f"📡 API Response: {api_response}")
        
        # Try to manually trigger loadModels
        print("🔄 Manually triggering loadModels...")
        manual_load_result = await page.evaluate('''async () => {
            if (window.trainingInterface && window.trainingInterface.loadModels) {
                try {
                    await window.trainingInterface.loadModels();
                    return { success: true };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            } else {
                return { success: false, error: 'TrainingInterface or loadModels not available' };
            }
        }''')
        
        print(f"🔧 Manual load result: {manual_load_result}")
        
        # Wait a moment and check options again
        await asyncio.sleep(2)
        
        options_after = await page.evaluate('''() => {
            const select = document.getElementById('test-model-select');
            if (!select) return null;
            return Array.from(select.options).map(opt => ({
                value: opt.value,
                text: opt.textContent
            }));
        }''')
        
        print(f"📋 Options after manual load: {len(options_after) if options_after else 0}")
        if options_after and len(options_after) > 1:
            print("✅ Models loaded successfully!")
            for i, opt in enumerate(options_after[:3]):
                print(f"  {i}: {opt['value']} - {opt['text']}")
        else:
            print("❌ Still no models in dropdown")
        
        # Click on the Testing tab to make sure we're in the right place
        print("🎯 Clicking Testing tab...")
        await page.click('[data-bs-target="#testing"]')
        await asyncio.sleep(1)
        
        # Try clicking on the dropdown to see if it triggers anything
        print("🖱️ Clicking on model dropdown...")
        await page.click('#test-model-select')
        await asyncio.sleep(1)
        
        # Check console for any errors
        print("\n⏳ Waiting 5 seconds for any additional logs...")
        await asyncio.sleep(5)
        
        print("🔍 Final DOM inspection...")
        final_check = await page.evaluate('''() => {
            const select = document.getElementById('test-model-select');
            if (!select) return { error: 'Element not found' };
            
            return {
                optionCount: select.options.length,
                innerHTML: select.innerHTML.substring(0, 500), // First 500 chars
                style: select.style.cssText,
                disabled: select.disabled,
                visible: select.offsetParent !== null
            };
        }''')
        
        print(f"📊 Final check: {final_check}")
        
        await browser.close()

if __name__ == '__main__':
    asyncio.run(test_dropdown_debug()) 