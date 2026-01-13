"""Verification script for clustering logic."""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
VIDEO_PATH = r"C:\Users\Gnana Prakash M\Downloads\Programs\Video Song ｜ Keladi Kannmani ｜ S P B ｜ Radhika ｜ Ilaiyaraaja Love Songs [033Z2WNg2Q.webm"


def wait_for_server():
    """Wait for API server to come online."""
    print("Waiting for server...")
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/health")
            print("Server is up!")
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    return False


def ingest_video():
    """Trigger ingestion for test video."""
    print(f"Ingesting: {VIDEO_PATH}")
    res = requests.post(
        f"{BASE_URL}/ingest",
        json={"path": VIDEO_PATH, "media_type_hint": "video"},
    )
    if res.status_code != 200:
        print(f"Ingest failed: {res.text}")
        sys.exit(1)
    return res.json()


def wait_for_job(file_path):
    """Poll for ingestion job completion."""
    print("Waiting for job completion...")
    while True:
        res = requests.get(f"{BASE_URL}/jobs")
        jobs = res.json().get("jobs", [])
        # Find job for our file
        # Note: file_path from ingest response might be normalized
        relevant_job = None
        for job in jobs:
            if Path(job["file_path"]).name == Path(VIDEO_PATH).name:
                relevant_job = job
                break

        if not relevant_job:
            print("Job not found yet...")
            time.sleep(2)
            continue

        status = relevant_job["status"]
        print(
            f"Job Status: {status} | Stage: {relevant_job.get('current_stage')} | Progress: {relevant_job.get('progress')}"
        )

        if status == "completed":
            print("Job Completed!")
            break
        if status == "failed":
            print(f"Job Failed: {relevant_job.get('error')}")
            sys.exit(1)

        time.sleep(3)


def trigger_clustering():
    """Trigger face and voice clustering manually."""
    print("Triggering Face Clustering...")
    res = requests.post(f"{BASE_URL}/faces/cluster")
    print("Face Cluster Result:", json.dumps(res.json(), indent=2))

    print("Triggering Voice Clustering...")
    res = requests.post(f"{BASE_URL}/voices/cluster")
    print("Voice Cluster Result:", json.dumps(res.json(), indent=2))


def get_clusters():
    """List resulting clusters."""
    print("Fetching Face Clusters...")
    res = requests.get(f"{BASE_URL}/faces/clusters")
    clusters = res.json().get("clusters", [])
    print(f"Found {len(clusters)} face clusters.")
    for c in clusters:
        print(
            f"  - Cluster {c['cluster_id']}: {c['face_count']} faces. Name: {c['name']}"
        )


if __name__ == "__main__":
    if not wait_for_server():
        print("Server failed to start.")
        sys.exit(1)

    # Check if already indexed to avoid re-ingesting if unnecessary (optional, but good for speed)
    # For verification, we might want to force re-ingest or just ingest.
    # The system allows re-ingestion.

    ingest_video()
    wait_for_job(VIDEO_PATH)
    trigger_clustering()
    get_clusters()
