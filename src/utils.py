import json
import pandas as pd
from pathlib import Path

def load_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (str or Path): Path to the JSON file.
        
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    try:
        return pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(data, file_path):
    """
    Save data to a JSON file.
    
    Args:
        data (list or pd.DataFrame): Data to save.
        file_path (str or Path): Path to save the file.
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
