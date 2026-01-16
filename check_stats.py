"""Utility script to verify Qdrant collection statistics."""
import requests

base_url = "http://localhost:6333"
collections = ["media_frames", "voice_segments", "faces", "scenes"]

print("--- Qdrant Collection Stats ---")
for col in collections:
    try:
        url = f"{base_url}/collections/{col}/points/count"
        resp = requests.post(url, json={"exact": True})
        if resp.status_code == 200:
            count = resp.json().get("result", {}).get("count", "N/A")
            print(f"{col}: {count}")
        else:
            print(f"{col}: Error {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"{col}: Failed - {e}")
