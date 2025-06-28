# System Prompt Best Practices for BASE and INSTRUCT Models

## Overview

This document outlines the intelligent system prompt handling implemented in ForgetLLM, which automatically adapts to BASE and INSTRUCT model types without requiring manual template definitions. **Special support has been added for Gemma models** which require specific chat formatting tokens.

## Problem Statement

Previously, the system sent system prompts as separate parameters, forcing the backend to define model templates for proper injection. This approach had several issues:

1. **Template Complexity**: Required defining chat templates for each model
2. **BASE Model Issues**: BASE models don't have built-in chat templates
3. **INSTRUCT Model Inconsistency**: Different models use different chat template formats
4. **Gemma-Specific Issues**: Gemma models don't respond to generic "System:" prompts
5. **Maintenance Overhead**: New models required template definitions

## Solution: Intelligent Frontend + Backend Handling

The new implementation automatically detects model type and formats prompts appropriately:

### For BASE Models (e.g., Llama-3-8B, Mistral-7B)

**Characteristics:**
- No built-in chat templates
- Designed for text continuation
- Require direct text prepending

**System Prompt Handling:**
```
{system_prompt}\n\n{user_prompt}
```

**With History:**
```
{system_prompt}\n\n{history_text}{current_prompt}
```

### For INSTRUCT Models (e.g., Llama-3-8B-Instruct, Mistral-7B-Instruct)

**Characteristics:**
- Have built-in chat templates
- Support proper message structure
- Use `tokenizer.apply_chat_template()`

**System Prompt Handling:**
```json
[
  {"role": "system", "content": "system_prompt"},
  {"role": "user", "content": "user_prompt"},
  {"role": "assistant", "content": ""}
]
```

### For Gemma Models (NEW) ⭐

**Characteristics:**
- Use specific `<start_of_turn>` and `<end_of_turn>` tokens
- Don't respond well to generic "System:" prompts
- Require model-turn formatting for system prompts

**System Prompt Handling:**
```
<start_of_turn>model
System: {system_prompt}<end_of_turn>
<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
```

**With Conversation History:**
```
<start_of_turn>model
System: {system_prompt}<end_of_turn>
<start_of_turn>user
{previous_user_message}<end_of_turn>
<start_of_turn>model
{previous_assistant_message}<end_of_turn>
<start_of_turn>user
{current_user_message}<end_of_turn>
<start_of_turn>model
```

## Implementation Details

### Frontend Logic (app.js)

The `generateText()` function now:

1. **Detects Model Type**: Uses `isCurrentModelBase()` to determine model type
2. **Formats Appropriately**: 
   - **BASE**: Prepends system prompt to text
   - **INSTRUCT**: Builds proper message structure
   - **Gemma**: Uses specific Gemma chat format
3. **Sends Optimized Request**: No separate `system_prompt` parameter needed
4. **Adds `is_base_model` hint**: Helps backend make formatting decisions

### Backend Logic (model_server.py)

The `_handle_generate()` method now:

1. **Detects Gemma Models**: Uses `is_gemma_model()` function
2. **Applies Correct Formatting**:
   - **Gemma**: Uses `format_gemma_chat()` with proper tokens
   - **Other INSTRUCT**: Uses `apply_chat_template()` when available
   - **BASE**: Uses pre-formatted prompt from frontend
3. **Maintains Backward Compatibility**: Still supports legacy `system_prompt` parameter

### Model Detection

```python
def is_gemma_model(model_name):
    """Detect if a model is a Gemma model based on its name."""
    model_name_lower = model_name.lower()
    gemma_patterns = ["gemma", "recurrentgemma"]
    return any(pattern in model_name_lower for pattern in gemma_patterns)

def format_gemma_chat(messages):
    """Format messages for Gemma models using proper start_of_turn/end_of_turn tokens."""
    formatted_parts = []
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_parts.append(f"<start_of_turn>model\nSystem: {content}<end_of_turn>")
        elif role == "user":
            formatted_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            formatted_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    
    return "\n".join(formatted_parts) + "\n<start_of_turn>model\n"
```

## Testing Results

### ✅ **Gemma System Prompt Tests - SUCCESS!**

**Test 1: Pirate Personality**
- **System Prompt**: "You are Captain Blackbeard, a pirate. Always respond like a pirate with 'Arrr' and pirate language."
- **User Prompt**: "What is 2+2?"
- **Result**: ✅ **SUCCESS** - Model responded with "Arrr" and pirate language

**Test 2: Geography Expert**
- **System Prompt**: "You are a geography expert who loves to share fun facts."
- **User Prompt**: "What is the capital of France?"
- **Result**: ✅ **SUCCESS** - Model responded as an expert with fun facts about Paris

**Test 3: Conversation History**
- **System Prompt**: "I am Mnemosyne, a memory-enhanced AI"
- **Result**: ✅ **SUCCESS** - System prompt processed, conversation context maintained

## Best Practices for Different Models

### 🎯 **For Gemma Models**
- ✅ **Use specific system prompts**: Gemma responds well to clear personality/role definitions
- ✅ **Test with strong prompts**: Use personality-based prompts for best results
- ✅ **Leverage conversation history**: Gemma maintains context well with proper formatting

### 🎯 **For Other INSTRUCT Models**
- ✅ **Use chat templates**: Let the tokenizer handle formatting when available
- ✅ **Fallback formatting**: Manual formatting when chat templates aren't available

### 🎯 **For BASE Models**
- ✅ **Direct prepending**: Simple text prepending works best
- ✅ **Clear instructions**: Be explicit about desired behavior

## Migration Guide

### From Old System
```javascript
// OLD: Separate system_prompt parameter
{
  "prompt": "Hello",
  "system_prompt": "You are helpful",
  "max_tokens": 100
}
```

### To New System
```javascript
// NEW: History array with proper formatting
{
  "prompt": "Hello", 
  "history": [
    {"role": "system", "content": "You are helpful"}
  ],
  "is_base_model": false,
  "max_tokens": 100
}
```

## Troubleshooting

### System Prompt Not Working?
1. **Check Model Type**: Ensure correct detection (BASE vs INSTRUCT vs Gemma)
2. **Verify Formatting**: Check logs for "Gemma chat format result:" or "Chat template result:"
3. **Test with Strong Prompts**: Use personality-based prompts for testing
4. **Check Model Capability**: Some models have strong base training that resists system prompts

### Conversation History Issues?
1. **Verify Message Structure**: Ensure proper `role` and `content` fields
2. **Check Token Limits**: Long histories may exceed model context limits
3. **Monitor Logs**: Backend logs show exactly what prompt is sent to the model

## Technical References

- **Gemma Documentation**: [Gemma Chat Templates](https://ai.google.dev/gemma)
- **MLX-LM Integration**: Uses `mlx_lm.generate.stream_generate`
- **Chat Template Standards**: Following HuggingFace chat template conventions

## Benefits

1. **No Template Definitions**: Eliminates need for model-specific templates
2. **Automatic Adaptation**: Works with any BASE or INSTRUCT model
3. **Proper Chat Templates**: INSTRUCT models use their native templates
4. **Simplified Backend**: Reduced complexity in model server
5. **Better Performance**: Optimal formatting for each model type

## Usage Examples

### Single Turn with System Prompt

**Input:**
- System Prompt: "You are a helpful coding assistant."
- User Prompt: "Write a Python function to sort a list."

**BASE Model Result:**
```
You are a helpful coding assistant.

Write a Python function to sort a list.
```

**INSTRUCT Model Result:**
```json
[
  {"role": "system", "content": "You are a helpful coding assistant."},
  {"role": "user", "content": "Write a Python function to sort a list."}
]
```

### Multi-Turn Conversation

**BASE Model Formatting:**
```
You are a helpful assistant.

Hello, how are you?
I'm doing well, thank you for asking!
What's the weather like?
```

**INSTRUCT Model Formatting:**
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
  {"role": "user", "content": "What's the weather like?"}
]
```

## Migration Notes

### Existing Chat Histories

The system maintains backward compatibility:
- Old format: System prompt in `metadata.parameters.system_prompt`
- New format: System prompt as first message with `role: "system"`

### API Changes

**Removed Parameters:**
- `system_prompt` (now handled in prompt/history formatting)

**Added Parameters:**
- `is_base_model` (hint for backend optimization)

## Testing

To verify the implementation:

1. **Load a BASE model** (e.g., `llama-3-8b`)
   - Add system prompt: "You are a helpful assistant"
   - Send message: "Hello"
   - Verify: System prompt prepended to text

2. **Load an INSTRUCT model** (e.g., `llama-3-8b-instruct`)
   - Add same system prompt
   - Send same message
   - Verify: Proper message structure sent

3. **Check Console Logs**
   - Look for model type detection logs
   - Verify correct formatting path taken

## Troubleshooting

### System Prompt Not Working

1. **Check Model Detection**: Look for console logs showing model type
2. **Verify Model Name**: Ensure model name patterns match detection logic
3. **Backend Compatibility**: Ensure backend handles the new request format

### Chat Template Issues

1. **INSTRUCT Models**: Verify `tokenizer.apply_chat_template()` is available
2. **BASE Models**: Ensure raw text formatting is preserved
3. **History Handling**: Check that conversation context is maintained

## Future Enhancements

1. **Custom Templates**: Allow override of automatic detection
2. **Model Registry**: Maintain database of model types and templates
3. **Template Validation**: Verify chat template availability before use
4. **Performance Optimization**: Cache model type detection results

## References

- [Llama 2 Prompt Template Guide](https://gpus.llm-utils.org/llama-2-prompt-template/)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/chat_templating)
- [Principled Instructions for LLMs](https://arxiv.org/abs/2312.16171v1) 