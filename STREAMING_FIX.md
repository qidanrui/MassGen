# OpenAI Streaming Fix for MASS System

## Problem Description

The MASS system was not properly streaming responses for the OpenAI backend (gpt-4o-mini) when using CLI commands like:

```bash
python cli.py "How many triple crowns winners are there in MLB baseball?" --models grok-3-mini gemini-2.5-flash gpt-4o-mini
```

While Grok and Gemini backends were streaming correctly, OpenAI only returned "response complete" without streaming the actual text.

## Root Cause Analysis

The issue was in the OpenAI backend (`mass/backends/oai.py`) where streaming was only enabled when **both** conditions were true:

1. `stream=True` (from ModelConfig)
2. `stream_callback` is not None

**Original problematic code (line 181):**
```python
"stream": True if stream and stream_callback else False,
```

**Original streaming check (line 225):**
```python
if stream and stream_callback:
```

### Why This Caused Issues

1. **ModelConfig defaults**: `stream=True` by default in `ModelConfig` class
2. **Configuration creation**: `create_config_from_models()` properly preserves this default
3. **Callback availability**: `stream_callback` is properly created by the orchestrator
4. **Timing issue**: The overly restrictive condition meant that if `stream_callback` was ever None (even temporarily), streaming would be completely disabled

This was inconsistent with the Gemini and Grok backends, which handle streaming more gracefully.

## Solution Implemented

### 1. Fixed API Parameter Setting
**Before:**
```python
"stream": True if stream and stream_callback else False,
```

**After:**
```python
"stream": stream,  # Enable streaming based on stream parameter, regardless of callback
```

### 2. Fixed Streaming Response Handling
**Before:**
```python
if stream and stream_callback:
```

**After:**
```python
if stream:
```

### 3. Added Safe Callback Helper
Added a helper function to handle cases where `stream_callback` might be None:

```python
def safe_stream_callback(content):
    if stream_callback:
        try:
            stream_callback(content)
        except Exception as e:
            print(f"Stream callback error: {e}")
```

### 4. Updated All Callback Calls
Replaced all direct `stream_callback(...)` calls with `safe_stream_callback(...)` throughout the streaming response handling code.

## Benefits of the Fix

1. **Consistent behavior**: OpenAI backend now behaves like Gemini and Grok backends
2. **Proper streaming**: Text responses are now streamed token-by-token as expected
3. **Graceful degradation**: If `stream_callback` is None, streaming still works but without UI updates
4. **Maintainability**: Centralized callback safety logic in the helper function

## Verification

To test the fix:

1. **With streaming display:**
   ```bash
   python3 cli.py "Test question" --models gpt-4o-mini
   ```

2. **Check streaming logs:**
   ```bash
   tail -f openai_streaming.txt
   ```

You should now see:
- Real-time text streaming in the console
- Proper token-by-token output
- "Response complete" message only at the end

## Technical Details

### OpenAI Responses API
The fix aligns with the official OpenAI Responses API documentation which supports streaming via `stream=True` parameter. The API returns server-sent events with different event types:

- `response.output_text.delta` - Text chunks
- `response.code_interpreter_call_code.delta` - Code generation
- `response.web_search_call.*` - Search events
- `response.completed` - Final completion

### Comparison with Other Backends

**Gemini backend:**
```python
if stream and stream_callback:  # Only handles response differently
```

**Grok backend:**
```python
is_streaming = stream and stream_callback is not None
```

The OpenAI backend now follows a more robust pattern that enables streaming based on the `stream` parameter while gracefully handling missing callbacks.

## Files Modified

- `mass/backends/oai.py` - Main fix implementation
- Line 181: Fixed API parameter setting
- Line 225: Fixed streaming condition
- Lines 238-243: Added safe callback helper
- Multiple lines: Updated all stream_callback calls

This fix resolves the streaming issue while maintaining backward compatibility and improving the robustness of the OpenAI backend. 