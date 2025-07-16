import os
import time
import json
from openai import OpenAI
from mock_tools import update_summary, check_updates

from dotenv import load_dotenv
load_dotenv()

def parse_completion(response, add_citations=True):
    """Parse the completion response from OpenAI API.
    
    Mainly three types of output in the response:
    - reasoning (no summary provided): ResponseReasoningItem(id='rs_6876b0d566d08198ab9f992e1911bd0a02ec808107751c1f', summary=[], type='reasoning', status=None)
    - web_search_call (actions: search, open_page, find_in_page): ResponseFunctionWebSearch(id='ws_6876b0e3b83081988d6cddd9770357c402ec808107751c1f', status='completed', type='web_search_call', action={'type': 'search', 'query': 'Economy of China Wikipedia GDP table'})
    - message: response output (including text and citations, optional)
    - code_interpreter_call (code provided): code output
    - function_call: function call, arguments, and name provided
    """
    text = ""
    code = []
    citations = []
    function_calls = []
    
    # Process the response output
    for r in response.output:
        if r.type == "message": # Final response, including text and citations (optional)
            for c in r.content:
                text += c.text
                if add_citations and hasattr(c, 'annotations') and c.annotations:
                    for annotation in c.annotations:
                        citations.append({
                            'url': annotation.url,
                            'title': annotation.title,
                            'start_index': annotation.start_index,
                            'end_index': annotation.end_index
                        })
        elif r.type == "code_interpreter_call":
            code.append(r.code)
        elif r.type == "web_search_call": 
            # detailed web search actions: search, open_page, find_in_page, etc
            pass
        elif r.type == "reasoning":
            # no summary provided for my current orginization
            pass
        elif r.type == "function_call":
            # tool output
            function_calls.append({"name": r.name, "arguments": r.arguments})
            pass
    
    # Add citations to text if available
    if add_citations and citations:
        # Sort citations by end_index in descending order to avoid shifting issues when inserting
        sorted_citations = sorted(citations, key=lambda c: c['end_index'], reverse=True)
        
        for idx, citation in enumerate(sorted_citations):
            end_index = citation['end_index']
            citation_link = f"[{len(citations) - idx}]({citation['url']})"
            text = text[:end_index] + citation_link + text[end_index:]
    
    return {"text": text, "code": code, "citations": citations, "function_calls": function_calls}

def process_message(messages, model="o4-mini", tools=["live_search", "code_execution"], max_retries=10, max_tokens=32000, temperature=0.5, top_p=1.0, api_key=None):
    """
    Generate content using OpenAI API.
    
    Args:
        messages: List of messages in OpenAI format
        model: The OpenAI model to use
        tools: List of tools to use
        max_retries: Maximum number of retry attempts
        max_tokens: Maximum number of tokens in response
        temperature: Temperature for generation
        top_p: Top-p value for generation
        api_key: OpenAI API key (if None, will get from environment)
    
    Returns:
        tuple: (text, code) where text is the generated text and code is list of executable code
    """
    # Get the API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Prepare tools
    formatted_tools = []
    
    # Add other custom tools
    for tool in tools:
        if isinstance(tool, dict):
            formatted_tools.append(tool)
        elif callable(tool):
            formatted_tools.append(function_to_json(tool))
        elif tool == "live_search": # built-in tools
            formatted_tools.append({"type": "web_search_preview"})
        elif tool == "code_execution": # built-in tools
            formatted_tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        else:
            raise ValueError(f"Invalid tool type: {type(tool)}")
    
    # Check if the model supports reasoning
    if "o1" in model or "o3" in model or "o4" in model:
        reasoning_model = True
    else:
        reasoning_model = False
    
    # Convert messages to the format expected by OpenAI responses API
    # For now, we'll use the last user message as input
    input_text = ""
    instructions = ""
    for message in reversed(messages):
        if message["role"] == "user":
            input_text = message["content"]
        if message["role"] == "system":
            instructions = message["content"]
    
    # print("--------------------------------")
    # print(f"[OAI] Instructions: {instructions}")
    # print(f"[OAI] Input: {input_text}")
    # _ = input("Press Enter to continue...")
    
    # Make API request with retry logic
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            response = client.responses.create(
                model=model,
                tools=formatted_tools if formatted_tools else None,
                instructions=instructions if instructions else None,
                input=input_text,
                reasoning={"effort": "low"} if reasoning_model else None,
                temperature=temperature if not reasoning_model else None,
                max_output_tokens=max_tokens,
                top_p=top_p
            )
            completion = response
            break
        except Exception as e:
            print(f"Error on attempt {retry + 1}: {e}")
            retry += 1
            time.sleep(1.5)

    if completion is None:
        raise Exception(f"Failed to get completion after {max_retries} retries")

    # print(completion)
    # output = completion.output
    # for o in output:
    #     print(f"-------------- {o.type} ------------------")
    #     print(o)
    #     _ = input("Press Enter to continue...")
    
    # Parse the completion and return text and code
    result = parse_completion(completion, add_citations=True)
    return result

# Example usage (you can remove this if not needed)
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": """
You are working with other agents to solve a task.
You can use search and code tools to help you solve the task.
During your task solving process, you should use the `update_summary` tool to progressively record your working process so that it can be shared with other agents.         
You should also use the `check_updates` tool to check other agents' current progress on the same task to help you find missing information or errors.
Remember: This is a collaborative effort. Work together, share insights and find missing information/errors, and build consensus toward the best solution.
"""
        },
        
        {"role": "user", "content": "Tell me the difference between the GDP of China and India in percentage for the past 20 years."
                 "Then use a suitable machine learning model to predict the GDP of China and India in 2035. Make sure to cite the sources of your information. "
                 "Record your final summary report by using the `update_summary` tool."}]
    
    from tools import update_summary, check_updates
    from util import function_to_json
    
    tools=["live_search", "code_execution"]
    tools.append(function_to_json(update_summary))
    tools.append(function_to_json(check_updates))
    
    result = process_message(messages, tools=tools, temperature=0.1)
    print("--------------------------------")
    print("Final response [text]: ", result["text"])
    print("Citations: ", result["citations"])
    print("All executed code: ", result["code"])
    print("All function calls: ", result["function_calls"])