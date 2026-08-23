import requests
import json
import sys

def test_query(question: str):
    url = "http://127.0.0.1:8000/api/v1/query"
    payload = {
        "question": question,
        "top_k": 3
    }
    
    print(f"Sending query: '{question}'...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("\n" + "="*50)
        print("LLM ANSWER:")
        print("="*50)
        print(data.get("answer", "No answer provided."))
        
        print("\n" + "="*50)
        print("SOURCES RETRIEVED:")
        print("="*50)
        sources = data.get("sources", [])
        if not sources:
            print("No sources found.")
        else:
            for i, src in enumerate(sources, 1):
                heading = src.get("heading", "Unknown Section")
                score = src.get("score", 0.0)
                url = src.get("url", "")
                ctype = src.get("content_type", "")
                print(f"[{i}] Score: {score:.3f} | Type: {ctype}")
                print(f"    Heading: {heading}")
                print(f"    URL: {url}")
                print("-" * 30)
                
        print(f"\nRetrieval time:  {data.get('retrieval_time_ms', 0):.1f} ms")
        print(f"Generation time: {data.get('generation_time_ms', 0):.1f} ms")
        print(f"Query transform: {data.get('transform_used') or 'Disabled'}")
        print(f"Reranker:        {data.get('reranker_used') or 'Disabled'}")
        print(f"Confidence:      {data.get('confidence', 'unknown')}")
        
    except Exception as e:
        print(f"Error querying the server: {e}")
        print("Make sure your uvicorn server is running on port 8000!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do I use sklearn config_context?"
        
    test_query(question)
