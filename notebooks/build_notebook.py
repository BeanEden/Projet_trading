
import json
import os
from pathlib import Path

def read_nb_cells(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            return nb.get('cells', [])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }

def main():
    root_dir = Path(".").resolve()
    notebooks_dir = root_dir / "notebooks"
    
    print(f"Building Master Notebook in {root_dir} from {notebooks_dir}...")

    master_cells = []

    # 1. Header & Setup
    master_cells.append(create_markdown_cell("# Master Trading Notebook: GBP/USD M15 Project\n\nThis notebook consolidates the entire workflow (T01-T11) into a single execution pipeline."))
    
    setup_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
import json
import pickle
from datetime import datetime
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Config
PROJECT_ROOT = Path('.').resolve()
DATA_DIR = PROJECT_ROOT / 'data'
M1_DIR = DATA_DIR 
M15_DIR = DATA_DIR / 'm15'
FEATURES_DIR = DATA_DIR / 'features'
MODELS_DIR = PROJECT_ROOT / 'models'

for d in [DATA_DIR, M15_DIR, FEATURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

YEARS = [2022, 2023, 2024]
plt.style.use('seaborn-v0_8')
"""
    master_cells.append(create_code_cell(setup_code))

    # 2. T01
    master_cells.append(create_markdown_cell("## 1. T01: Import M1 Data"))
    t01_cells = read_nb_cells(notebooks_dir / "T01_Import_M1_Controle_Regularite.ipynb")
    # Filter out redundant imports or markdown if needed, but keeping primarily code logic
    master_cells.extend(t01_cells[2:]) # Skip header cells usually

    # 3. T02 (Recreated logic since it's not a standalone file)
    master_cells.append(create_markdown_cell("## 2. T02: Aggregation M1 -> M15"))
    t02_code = """
def aggregate_m1_to_m15(df_m1):
    if df_m1 is None or df_m1.empty:
        return None
    
    # Resample logic
    df_m15 = df_m1.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Tick counte
    df_m15['tick_count'] = df_m1['close'].resample('15T').count()
    
    df_m15.dropna(inplace=True)
    return df_m15

dfs_m15 = {}
# Assuming dfs_m1 was created in T01 cells. If not, we might need to adjust.
# We will wrap in try-except to avoid breaking if T01 variables aren't strictly 'dfs_m1'
try:
    if 'dfs_m1' in locals():
        for year, df_m1 in dfs_m1.items():
            print(f"Aggregating {year}...")
            df_15 = aggregate_m1_to_m15(df_m1)
            dfs_m15[year] = df_15
            
            # Save
            out_path = M15_DIR / f"GBPUSD_M15_{year}.csv"
            df_15.to_csv(out_path)
            print(f"Saved {out_path}")
except Exception as e:
    print(f"Aggregation step skipped or failed: {e}")
"""
    master_cells.append(create_code_cell(t02_code))

    # 4. T03
    master_cells.append(create_markdown_cell("## 3. T03: Cleaning"))
    t03_cells = read_nb_cells(notebooks_dir / "T03_Nettoyage_M15.ipynb")
    master_cells.extend(t03_cells[2:])

    # 5. T05
    master_cells.append(create_markdown_cell("## 4. T05: Feature Engineering"))
    t05_cells = read_nb_cells(notebooks_dir / "T05_Feature_Engineering.ipynb")
    master_cells.extend(t05_cells[2:])

    # 6. T07
    master_cells.append(create_markdown_cell("## 5. T07: Machine Learning"))
    t07_cells = read_nb_cells(notebooks_dir / "T07_Machine_Learning.ipynb")
    master_cells.extend(t07_cells[2:])

    # 7. T09
    master_cells.append(create_markdown_cell("## 6. T09: Comparative Evaluation"))
    t09_cells = read_nb_cells(notebooks_dir / "T09_Evaluation_Comparative.ipynb")
    master_cells.extend(t09_cells[2:]) 

    # 8. T11
    master_cells.append(create_markdown_cell("## 7. T11: Versioning"))
    t11_cells = read_nb_cells(notebooks_dir / "T11_Versioning_Modele.ipynb")
    master_cells.extend(t11_cells[2:])

    # Write Master Notebook
    nb_content = {
        "cells": master_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = root_dir / "Master_Trading_Notebook.ipynb"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb_content, f, indent=1)
    
    print(f"Successfully created {out_file} with {len(master_cells)} cells.")

if __name__ == "__main__":
    main()
