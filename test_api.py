import urllib.request
import json
import time

def run_tests():
    print("Test 1: Settings endpoint")
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/settings")
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=4))
        
    print("\nTest 2: Ollama models endpoint")
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/ollama/models")
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=4))
        
    print("\nTest 3: Query with use_query_transform=false")
    data = json.dumps({"question": "what is this project about", "use_query_transform": False}).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/query", data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            print("Status:", resp.status)
            res = json.loads(resp.read())
            print("Transform Used:", res.get("transform_used"))
    except Exception as e:
        print("Error:", e)

    print("\nTest 4: Query with use_query_transform=true")
    data = json.dumps({"question": "what is this project about", "use_query_transform": True}).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/query", data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            print("Status:", resp.status)
            res = json.loads(resp.read())
            print("Transform Used:", res.get("transform_used"))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    run_tests()
