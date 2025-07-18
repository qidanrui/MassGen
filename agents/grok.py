import os
import time
import json
import threading
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, assistant, system
from xai_sdk.search import SearchParameters

load_dotenv()

def parse_completion(response, add_citations=True):
    """Parse the completion response from Grok API."""
    text = response.content
    code = []
    citations = []
    
    if add_citations and hasattr(response, 'citations') and response.citations:
        citations = []
        for citation in response.citations:
            citations.append({
                'url': citation,
                'title': '',
                'start_index': -1,
                'end_index': -1
            })
        
        if citations:
            citation_content = []
            for idx, citation in enumerate(citations):
                citation_content.append(f"[{idx}]({citation['url']})")
            text = text + "\n\n" + "\n".join(citation_content)
    
    return {"text": text, "code": code, "citations": citations, "function_calls": []}

def process_message(messages, model="grok-4", tools=["live_search"], max_retries=10, max_tokens=32000, temperature=None, top_p=None, api_key=None, processing_timeout=180, stream=False, stream_callback=None):
    """
    Generate content using Grok API with optional streaming support.
    """
    def do_inference():
        """Internal function that contains all the processing logic."""
        if api_key is None:
            api_key_val = os.getenv("XAI_API_KEY")
        else:
            api_key_val = api_key

        if not api_key_val:
            raise ValueError("XAI_API_KEY not found in environment variables")

        client = Client(api_key=api_key_val)

        search_parameters = None
        if "live_search" in tools:
            search_parameters = SearchParameters(
                mode="on",
                return_citations=True,
            )
            # Notify streaming callback about search being performed
            if stream and stream_callback is not None:
                try:
                    # According to XAI SDK docs, search queries are not available in streaming chunks
                    # We can only notify that search is being performed
                    stream_callback("[SEARCH] Performing web search...")
                except Exception as e:
                    print(f"Stream callback error: {e}")

        def make_grok_request(stream=False):
            chat = client.chat.create(
                model=model,
                search_parameters=search_parameters,
                temperature=temperature if temperature else None,
                top_p=top_p if top_p else None,
                max_tokens=max_tokens if max_tokens else None,
            )
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    chat.append(system(content))
                elif role == "user":
                    chat.append(user(content))
                elif role == "assistant":
                    chat.append(assistant(content))
            
            if stream:
                return chat.stream()
            else:
                return chat.sample()

        completion = None
        retry = 0
        while retry < max_retries:
            try:
                is_streaming = stream and stream_callback is not None
                completion = make_grok_request(stream=is_streaming)
                break
            except Exception as e:
                print(f"Error on attempt {retry + 1}: {e}")
                retry += 1
                time.sleep(1.5)

        if completion is None:
            print(f"Failed to get completion after {max_retries} retries, returning empty response")
            return {"text": "", "code": [], "citations": [], "function_calls": []}

        if stream and stream_callback is not None:
            text = ""
            code = []
            citations = []
            function_calls = []
            
            try:
                for response, chunk in completion:
                    # Handle XAI SDK streaming format: (response, chunk)
                    # According to docs: chunk contains text deltas, response accumulates
                    delta_content = None
                    
                    # Extract delta content from chunk
                    if hasattr(chunk, 'content') and chunk.content:
                        delta_content = chunk.content
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'content') and choice.content:
                            delta_content = choice.content
                        elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                            delta_content = choice.delta.content
                    elif hasattr(chunk, 'text'):
                        delta_content = chunk.text
                    
                    if delta_content:
                        text += delta_content
                        try:
                            stream_callback(delta_content)
                        except Exception as e:
                            print(f"Stream callback error: {e}")
                    
                    # Check if this is the final chunk with citations
                    if hasattr(response, 'citations') and response.citations:
                        citations = []
                        for citation in response.citations:
                            citations.append({
                                'url': citation,
                                'title': '',
                                'start_index': -1,
                                'end_index': -1
                            })
                        
                        # Notify about found sources
                        if stream_callback is not None:
                            try:
                                stream_callback(f"[SEARCH] Found {len(citations)} web sources")
                            except Exception as e:
                                print(f"Stream callback error: {e}")
                        
            except Exception as e:
                print(f"Streaming error: {e}")
                # Fall back to non-streaming
                completion = make_grok_request(stream=False)
                result = parse_completion(completion, add_citations=True)
                return result

            # Add search completion message
            if stream_callback is not None:
                try:
                    stream_callback("[SEARCH] Web search completed")
                except Exception as e:
                    print(f"Stream callback error: {e}")
            
            result = {"text": text, "code": code, "citations": citations, "function_calls": function_calls}
        else:
            result = parse_completion(completion, add_citations=True)
        
        return result

    result_container = {"result": None, "completed": False}
    
    def thread_worker():
        try:
            result_container["result"] = do_inference()
        except Exception as e:
            print(f"Error in thread worker: {e}")
            result_container["result"] = {"text": "", "code": [], "citations": [], "function_calls": []}
        finally:
            result_container["completed"] = True
    
    worker_thread = threading.Thread(target=thread_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=processing_timeout)
    
    if result_container["completed"]:
        return result_container["result"]
    else:
        print(f"⏰ SYSTEM: Processing timed out after {processing_timeout} seconds")
        return {"text": "", "code": [], "citations": [], "function_calls": []}

if __name__ == '__main__':
    # Test streaming callback functionality without making API calls
    def test_stream_callback():
        chunks_received = []
        
        def stream_handler(chunk):
            chunks_received.append(chunk)
            print(f"Received chunk: {chunk}")
        
        # Test the callback function works
        stream_handler("test chunk")
        assert len(chunks_received) == 1
        assert chunks_received[0] == "test chunk"
        print("✓ Stream callback test passed")
    
    test_stream_callback()
    
    # Cheap real API test with minimal tokens
    messages = [
        {"role": "user", "content": "Hi"}
    ]
    
    def stream_handler(chunk):
        print(chunk, end="")

    # Uncomment below to test streaming with tools (costs money)
    # print("--- Streaming Test with Tools ---")
    # messages_with_search = [
    #     {"role": "user", "content": "What's the latest news about AI?"}
    # ]
    # result = process_message(messages_with_search, model="grok-3", tools=["live_search"], stream=True, stream_callback=stream_handler, max_tokens=50)
    # print(f"\nResult: {result}")
    # print("Test completed!")