import json
import sys
import time

try:
    import requests
except ImportError:
    print("This script requires the 'requests' library.")
    print("Please install it running: pip install requests")
    sys.exit(1)

SERVER_URL = "http://localhost:8000"

def test_generate_endpoint(prompt: str):
    print("=" * 70)
    print(f"Testing CUSTOM STREAMING Endpoint: {SERVER_URL}/generate")
    print(f"Prompt: {prompt}")
    print("-" * 70)
    
    start_time = time.time()
    payload = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        # Use stream=True to tell requests to keep the connection open for streaming
        response = requests.post(f"{SERVER_URL}/generate", json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"\nError: Received status code {response.status_code}")
            print(response.text)
            return

        # Stream chunk-by-chunk and output in real-time
        print("Response stream: ", end="", flush=True)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
                
        print(f"\n\nStream finished in {time.time() - start_time:.2f} seconds!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the LLM Server.")
        print("Please ensure the server is running on port 8000.")
        print("=" * 70)

def test_chat_completions_endpoint(prompt: str):
    print("\n" + "=" * 70)
    print(f"Testing OPENAI-COMPATIBLE Endpoint: {SERVER_URL}/v1/chat/completions")
    print(f"User Message: {prompt}")
    print("-" * 70)
    
    start_time = time.time()
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful and concise coding assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": True
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"\nError: Received status code {response.status_code}")
            print(response.text)
            return

        print("Response delta stream: ", end="", flush=True)
        
        # Read the lines of the SSE stream
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
                
            # SSE format is "data: <content>"
            if line.startswith("data: "):
                data_content = line[6:].strip()
                
                # Check for stream end
                if data_content == "[DONE]":
                    break
                    
                try:
                    data_json = json.loads(data_content)
                    choices = data_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    # If it's not JSON (e.g. error messages), print raw line
                    print(f"\nRaw SSE Data: {data_content}")
                    
        print(f"\n\nStream finished in {time.time() - start_time:.2f} seconds!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the LLM Server.")
        print("Please ensure the server is running on port 8000.")
        print("=" * 70)

if __name__ == "__main__":
    test_prompt = "Explain in one sentence why learning Python is useful."
    
    # Run the tests
    test_generate_endpoint(test_prompt)
    test_chat_completions_endpoint(test_prompt)
