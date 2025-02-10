import os
import pandas as pd


def load_dataset(data_path, data_type='train'):
    """Load all data files for either train or test set"""
    file_map = {
        'sepsis_labels': f"SepsisLabel_{data_type}.csv",
        'devices': f"devices_{data_type}.csv",
        'drugs': f"drugsexposure_{data_type}.csv",
        'lab_measurements': f"measurement_lab_{data_type}.csv",
        'meds_measurements': f"measurement_meds_{data_type}.csv",
        'observations': f"measurement_observation_{data_type}.csv",
        'demographics': f"person_demographics_episode_{data_type}.csv",
        'procedures': f"proceduresoccurrences_{data_type}.csv"
    }
    
    data_dict = {}
    
    for key, filename in file_map.items():
        file_path = os.path.join(data_path, filename)
        try:
            data_dict[key] = pd.read_csv(file_path)
            print(f"Loaded {filename} successfully")
        except FileNotFoundError:
            print(f"Warning: {filename} not found in {data_path}")
            data_dict[key] = None
    
    return data_dict