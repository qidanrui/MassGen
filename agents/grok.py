import os
import time
import json
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
    
    # Extract citations if available
    if add_citations and hasattr(response, 'citations') and response.citations:
        citations = []
        # citations: ['https://www.democracynow.org/2025/7/9/headlines', 'https://www.thehindu.com/news/top-news-of-the-day-july-9-2025/article69791894.ece', 'https://abcnews.go.com/International', 'https://www.news9.com/story/686cfcb257d1953b20e74582/6-things-to-know', 'https://www.npr.org/programs/all-things-considered/2025/07/09/all-things-considered-for-july-09-2025', 'https://www.pbs.org/newshour/show/july-9-2025-pbs-news-hour-full-episode', 'https://telanganatoday.com/cartoon-today-on-july-12-2025', 'https://www.hindustantimes.com/astrology/horoscope/horoscope-today-july-9-2025-3-zodiac-signs-are-about-to-get-lucky-today-101752070899083.html', 'https://www.nbcnews.com/now/video/why-july-9-2025-is-one-of-the-shortest-days-in-recorded-history-242969157656', 'https://www.nbcnews.com/nightly-news-netcast/video/nightly-news-full-broadcast-july-9th-242971205754', 'https://x.com/sidhant/status/1939759253486805484', 'https://x.com/nordiqa42645/status/1939502791610744869', 'https://x.com/ankitatIIMA/status/1941562169314902523', 'https://x.com/sidhant/status/1935318297001742412', 'https://x.com/Zlatti_71/status/1941043566710497319']
        for citation in response.citations:
            citations.append({
                'url': citation,
                'title': '', # no title for grok
                'start_index': -1, # no start index for grok
                'end_index': -1 # no end index for grok
            })
        
        # Add citation links to text if available
        if citations:
            citation_content = []
            for idx, citation in enumerate(citations):
                citation_content.append(f"[{idx}]({citation['url']})")
            text = text + "\n\n" + "\n".join(citation_content)
    
    return {"text": text, "code": code, "citations": citations}

def process_message(messages, model="grok-4", tools=["live_search"], max_retries=10, max_tokens=32000, temperature=0.7, top_p=1.0, api_key=None):
    """
    Generate content using Grok API.
    
    Args:
        messages: List of messages in OpenAI format
        model: The Grok model to use
        tools: List of tools to use
        max_retries: Maximum number of retry attempts
        max_tokens: Maximum number of tokens in response
        temperature: Temperature for generation
        top_p: Top-p value for generation
        api_key: Grok API key (if None, will get from environment)
    
    Returns:
        dict: {"text": text, "code": code, "citations": citations}
    """
    # Get the API key
    if api_key is None:
        api_key = os.getenv("XAI_API_KEY")
    
    if not api_key:
        raise ValueError("XAI_API_KEY not found in environment variables")

    # Create Grok client
    client = Client(api_key=api_key)
    
    # Set up search parameters
    search_parameters = None
    if "live_search" in tools:
        search_parameters = SearchParameters(
            mode="on",
            return_citations=True,
        )
    if "code_execution" in tools:
        pass # Grok does not have code execution built-in
    # TODO: add other tools
    
    # Make API request with retry logic
    completion = None
    retry = 0
    while retry < max_retries:
        try:
            # Create chat instance
            chat = client.chat.create(
                model=model,
                search_parameters=search_parameters,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            # Convert messages from OpenAI format to Grok format
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    chat.append(system(content))
                elif role == "user":
                    chat.append(user(content))
                elif role == "assistant":
                    chat.append(assistant(content))
            
            # Get response
            response = chat.sample()
            completion = response
            break
            
        except Exception as e:
            print(f"Error on attempt {retry + 1}: {e}")
            retry += 1
            time.sleep(1.5)

    if completion is None:
        raise Exception(f"Failed to get completion after {max_retries} retries")

    # Parse the completion and return text, code, and citations
    result = parse_completion(completion, add_citations=True)
    return result

# Example usage (you can remove this if not needed)
if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Which are the three longest rivers mentioned in the Aeneid?"}]
    
    result = process_message(messages, tools=["live_search"])
    print("##########Response#############")
    print(result["text"])
    print("##########Code#############")
    print(result["code"])
    print("##########Citations#############")
    print(result["citations"])
