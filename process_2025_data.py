import pandas as pd
import numpy as np
import os
import sys

# Paths
INPUT_DIR = "HISTDATA_COM_MT_GBPUSD_M12025"
DEFAULT_INPUT_FILE = os.path.join(INPUT_DIR, "DAT_MT_GBPUSD_M1_2025.csv")
OUTPUT_FILE = "data/m15/GBPUSD_M15_2025.csv"

def find_input_file():
    if os.path.exists(DEFAULT_INPUT_FILE):
        return DEFAULT_INPUT_FILE
    
    print(f"Warning: {DEFAULT_INPUT_FILE} not found. Searching in {INPUT_DIR}...")
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".csv"):
                found_path = os.path.join(root, file)
                print(f"Found CSV at: {found_path}")
                return found_path
    return None

def process_data():
    input_file = find_input_file()
    if not input_file:
        print(f"Error: Could not find any .csv file in {INPUT_DIR}")
        return

    print(f"Reading {input_file}...")
    try:
        # Read without header, assign names
        df = pd.read_csv(input_file, names=["date", "time", "open", "high", "low", "close", "volume"], header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Parsing timestamps...")
    # Combine Date & Time
    if not all(col in df.columns for col in ["date", "time"]):
        # Maybe it has a header? Let's check first row
        print("Columns found:", df.columns)
        print("First row:", df.iloc[0].values)
        return

    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df.set_index("timestamp", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)

    print("Resampling to 15min...")
    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    # Resample
    df_15m = df.resample("15min").agg(ohlc_dict).dropna()
    
    # Check if empty
    if df_15m.empty:
        print("Error: Resampled DataFrame is empty.")
        return

    # Rename
    df_15m.rename(columns={
        "open": "open_15m",
        "high": "high_15m",
        "low": "low_15m",
        "close": "close_15m"
    }, inplace=True)
    
    # Tick count
    df_15m["tick_count"] = df_15m["volume"].astype(int) 

    # Reorder
    df_15m = df_15m[["open_15m", "high_15m", "low_15m", "close_15m", "volume", "tick_count"]]
    df_15m.reset_index(inplace=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    df_15m.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    process_data()
