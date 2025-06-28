# Model Switching Fix

## Issue Description
When changing models in the UI, the backend was not properly switching to the new model. Users would select a different model but the system would continue generating text using the previous model.

## Root Causes Identified

### 1. Parameter Mismatch (Primary Cause - FIXED)
The frontend JavaScript in `forgellm/web/static/app.js` was sending incorrect parameter names to the API:

**Before (Incorrect):**
```javascript
body: JSON.stringify({
    model: model,           // ❌ Wrong parameter name
    adapter: adapter,       // ❌ Wrong parameter name  
    system_prompt: prompt
})
```

**After (Fixed):**
```javascript
body: JSON.stringify({
    model_name: model,      // ✅ Correct parameter name
    adapter_path: adapter,  // ✅ Correct parameter name
    system_prompt: prompt
})
```

### 2. Model Server Memory Management (Secondary Cause - FIXED)
The model server was not properly unloading previous models before loading new ones, which could cause memory issues and state conflicts.

**Fixed by adding proper cleanup in `model_server.py`:**
```python
# Unload previous model first to free memory
if MODEL is not None:
    logger.info("Unloading previous model to free memory")
    MODEL = None
    TOKENIZER = None
    # Force garbage collection to free GPU memory
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
        logger.info("Cleared MLX metal cache")
    except:
        pass
```

## Testing
Created `test_model_switching.py` to verify the fix works correctly. The test:
1. Loads different models sequentially
2. Generates text with each model
3. Verifies the correct model is actually loaded
4. Confirms different models produce different outputs

## Files Modified
1. `forgellm/web/static/app.js` - Fixed parameter names in model loading request
2. `model_server.py` - Added proper model cleanup before loading new models
3. `test_model_switching.py` - Created test script to verify the fix

## Verification
The fix has been tested and confirmed to work correctly. Model switching now properly changes the backend model and generates text using the newly selected model. 