import os
import json

models_dir = "models"
corrupted_files = []

for root, dirs, files in os.walk(models_dir):
    if "eval_results.json" in files:
        file_path = os.path.join(root, "eval_results.json")
        try:
            with open(file_path, "r") as f:
                json.load(f)
        except Exception as e:
            print(f"CORRUPTED: {file_path} - {e}")
            corrupted_files.append(file_path)

if not corrupted_files:
    print("No corrupted files found in models/ subdirectory.")
