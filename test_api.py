import requests
import json

try:
    r = requests.get("http://127.0.0.1:5000/api/models_comparison")
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Data keys: {list(data.keys())}")
        # Print a snippet of one model's data
        if data:
            first_model = list(data.keys())[0]
            print(f"Sample data for {first_model}: {json.dumps(data[first_model], indent=2)[:500]}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Error connecting to dashboard: {e}")
