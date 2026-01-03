import sys
import importlib

def check_import(module_name, alias=None):
    try:
        importlib.import_module(module_name)
        print(f" [x] {alias or module_name}")
    except ImportError as e:
        print(f" [ ] {alias or module_name} ({e})")

def main():
    print("Checking critical dependencies...")
    check_import("google.generativeai")
    check_import("scenedetect")
    check_import("cv2")
    check_import("hdbscan")
    check_import("a2a")
    check_import("sentence_transformers")
    
    print("\nChecking project modules...")
    try:
        from core.storage.db import VectorDB
        print(" [x] core.storage.db")
    except Exception as e:
        print(f" [ ] core.storage.db ({e})")

    try:
        from api.server import app
        print(" [x] api.server")
    except Exception as e:
        print(f" [ ] api.server ({e})")

if __name__ == "__main__":
    main()
