import shutil
import os
import sys
import time

path = r"c:\Users\ludop\OneDrive\Documents\Cours\Sup de Vinci\Datascience\Projet_Final\models\v11"

print(f"Attempting to delete: {path}")

if not os.path.exists(path):
    print("Path does not exist.")
    sys.exit(0)

try:
    shutil.rmtree(path)
    print("Successfully deleted.")
except Exception as e:
    print(f"Error deleting: {e}")
    # Try to rename if delete fails
    try:
        new_path = path + "_trash_" + str(int(time.time()))
        os.rename(path, new_path)
        print(f"Renamed to {new_path} instead.")
    except Exception as e2:
        print(f"Error renaming: {e2}")
