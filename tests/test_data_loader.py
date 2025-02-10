import os
import pandas as pd
from src.data_loader import load_dataset

def test_load_dataset():
    # Mock paths for training and testing data
    train_path = "data/training_data"
    test_path = "data/testing_data"

    # Test loading training data
    train_data = load_dataset(train_path, data_type="train")
    assert isinstance(train_data, dict), "Training data should be a dictionary."
    assert "sepsis_labels" in train_data, "Sepsis labels should be loaded."
    assert train_data["sepsis_labels"] is not None, "Sepsis labels should not be None."

    # Test loading testing data
    test_data = load_dataset(test_path, data_type="test")
    assert isinstance(test_data, dict), "Testing data should be a dictionary."
    assert "sepsis_labels" in test_data, "Sepsis labels should be loaded (even if empty for test set)."
    assert test_data["sepsis_labels"] is not None or test_data["sepsis_labels"].empty, "Sepsis labels can be empty for test set."

