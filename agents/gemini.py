import os
import time
import requests
import json

from dotenv import load_dotenv
import os
load_dotenv()

def parse_completion(completion, add_citations=True):
    """Parse the completion response from Gemini API."""
    candidates = completion["candidates"]
    candidate = candidates[0]
    
    # print("candidate: ", json.dumps(candidate, indent=4))

    # The final response is in the content field
    parts = candidate["content"]["parts"]
    
    text = ""
    code = []
    citations = []
    
    # Parse the text and code from the parts
    for part in parts:
        # print("part: ", json.dumps(part, indent=4))
        if "text" in part:
            text += part["text"]
        elif "executableCode" in part:
            code.append(part["executableCode"]["code"])

    # The grounding metadata is in the groundingMetadata field
    groundingMetadata = candidate["groundingMetadata"]
    
    # Extract citations if available
    if add_citations and "groundingSupports" in groundingMetadata and "groundingChunks" in groundingMetadata:
        supports = groundingMetadata["groundingSupports"]
        chunks = groundingMetadata["groundingChunks"]

        # First, collect all citation information
        for support in supports:
            start_index = support["segment"]["startIndex"]
            end_index = support["segment"]["endIndex"]
            if support["groundingChunkIndices"]:
                for i in support["groundingChunkIndices"]:
                    if i < len(chunks):
                        chunk = chunks[i]
                        if "web" in chunk:
                            uri = chunk["web"]["uri"]
                            title = chunk["web"].get("title", "")
                            citations.append({
                                'url': uri,
                                'title': title,
                                'start_index': start_index,
                                'end_index': end_index,
                                'chunk_index': i
                            })

        # Sort supports by end_index in descending order to avoid shifting issues when inserting.
        sorted_supports = sorted(supports, key=lambda s: s["segment"]["endIndex"], reverse=True)

        for support in sorted_supports:
            end_index = support["segment"]["endIndex"]
            if support["groundingChunkIndices"]:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in support["groundingChunkIndices"]:
                    if i < len(chunks):
                        uri = chunks[i]["web"]["uri"]
                        citation_links.append(f"[{i + 1}]({uri})")

                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

    return {"text": text, "code": code, "citations": citations}

def process_message(messages, model="gemini-2.5-flash", tools=["live_search", "code_execution"], max_retries=10, max_tokens=None, temperature=None, top_p=None, api_key=None):
    """
    Generate content using Gemini API.
    
    Args:
        messages: List of messages in OpenAI format
        model: The Gemini model to use
        tools: List of tools to use
        max_retries: Maximum number of retry attempts
        max_tokens: Maximum number of tokens in response
        temperature: Temperature for generation
        top_p: Top-p value for generation
        api_key: Gemini API key (if None, will get from environment)
    
    Returns:
        tuple: (text, code, citations) where text is the generated text, code is list of executable code, and citations is list of citation metadata
    """
    # Get the API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    # Safety settings
    gemini_safety_settings = [
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    # Generation config
    gemini_generation_config = {
            "candidateCount": 1,
        }
    if max_tokens:
        gemini_generation_config["maxOutputTokens"] = max_tokens
    if temperature:
        gemini_generation_config["temperature"] = temperature
    if top_p:
        gemini_generation_config["topP"] = top_p

    # Convert messages from OpenAI format to Gemini format
    gemini_messages = []
    used_tools = {}
    sys_prompt = ""
    for msg_idx, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            sys_prompt = message["content"]
        elif role == "user":
            parts = [{"text": message["content"]}]
            gemini_messages.append({"role": "user", "parts": parts})
        elif role in ["assistant", "model"]:
            if message.get("tool_calls", None):
                function_calls = []
                for tool_call in message["tool_calls"]:
                    function_calls.append({"functionCall": {"name": tool_call.get("function").get("name"), 
                                                            "args": json.loads(tool_call.get("function").get("arguments"))}})
                    used_tools[tool_call.get("id")] = tool_call.get("function").get("name")
                gemini_messages.append(
                    {"role": "model", "parts": function_calls}
                )
            else:
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": message["content"]}]}
                )
        elif role == "tool":
            gemini_messages.append(
                {"role": "user", "parts": [{
                    "functionResponse": {
                        "name": used_tools[message["tool_call_id"]],
                        "response": {
                            "name": used_tools[message["tool_call_id"]],
                            "content": message["content"]
                        }
                    }
                }]}
            )

    # System instruction (optional)
    if sys_prompt:
        gemini_system_instruction = {
            "parts": [{"text": sys_prompt}]
        }
    else:
        gemini_system_instruction = None
        
    # Tools (optional)
    gemini_tools = []
    # The two built-in tools for gemini
    if "live_search" in tools:
        gemini_tools.append({
            "google_search": {}
        })
    if "code_execution" in tools:
        gemini_tools.append({
            "code_execution": {}
        })
    # TODO: add other custom tools

    # Construct the payload according to the version
    payload = {
        "contents": gemini_messages,
        "safetySettings": gemini_safety_settings,
        "generationConfig": gemini_generation_config,
    }
    if gemini_system_instruction:
        payload["system_instruction"] = gemini_system_instruction
    if gemini_tools:
        payload["tools"] = gemini_tools
    
    headers = {"Content-Type": "application/json"}
    beta = "v1alpha" if "thinking" in model else "v1beta"
    url = f"https://generativelanguage.googleapis.com/{beta}/models/{model}:generateContent?key={api_key}"
    
    # Debug
    # print(f"[GEMINI] Payload: {json.dumps(payload, indent=4)}")
    # _ = input("[GEMINI] Press Enter to continue...")
    
    # Make API request
    # Retry up to max_retries times if there is an error
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            response = requests.post(url, headers=headers, json=payload)
            # response.raise_for_status()
            completion = response.json()
            # print("completion: ", completion)
            break
        except Exception as e:
            print(e)
            retry += 1
            time.sleep(1.5)

    if completion is None:
        raise Exception(f"Failed to get completion after {max_retries} retries")

    # Parse the completion and return text, code, and citations
    result = parse_completion(completion, add_citations=True)
    
    # print(json.dumps(result, indent=4))
    # _ = input("[GEMINI] Press Enter to continue...")
    return result

# Example usage (you can remove this if not needed)
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Which are the three longest rivers mentioned in the Aeneid?"}]
    
    result = process_message(messages, tools=["live_search", "code_execution"])
    print("text (with citations): ", result["text"])
    print("code: ", result["code"])
    print("citations: ", result["citations"])