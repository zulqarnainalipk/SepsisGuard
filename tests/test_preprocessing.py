import pandas as pd
from src.preprocessing import preprocess_dataset

def test_preprocess_dataset():
    # Mock sepsis labels
    sepsis_labels_train = pd.DataFrame({
        "person_id": [1, 1, 2],
        "measurement_datetime": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 10:00:00"],
        "SepsisLabel": [0, 1, 0]
    })

    sepsis_labels_test = pd.DataFrame({
        "person_id": [3, 3],
        "measurement_datetime": ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
    })

    # Mock demographics
    demographics = pd.DataFrame({
        "person_id": [1, 2, 3],
        "birth_datetime": ["2010-01-01", "2012-01-01", "2015-01-01"]
    })

    # Preprocess training data
    train_df, encoders = preprocess_dataset(
        sepsis_labels=sepsis_labels_train,
        devices=None,
        drugs=None,
        lab_measurements=None,
        meds_measurements=None,
        observations=None,
        demographics=demographics,
        procedures=None,
        is_train=True
    )

    # Assertions for training data
    assert isinstance(train_df, pd.DataFrame), "Processed training data should be a DataFrame."
    assert "age_in_months" in train_df.columns, "Age in months should be calculated."
    assert "SepsisLabel" in train_df.columns, "Sepsis label should be present."
    assert train_df["age_in_months"].iloc[0] > 0, "Age in months should be greater than 0."
    assert encoders is not None, "Encoders should be returned for training data."

    # Preprocess testing data
    test_df = preprocess_dataset(
        sepsis_labels=sepsis_labels_test,
        devices=None,
        drugs=None,
        lab_measurements=None,
        meds_measurements=None,
        observations=None,
        demographics=demographics,
        procedures=None,
        is_train=False,
        encoders=encoders
    )

    # Assertions for testing data
    assert isinstance(test_df, pd.DataFrame), "Processed testing data should be a DataFrame."
    assert "age_in_months" in test_df.columns, "Age in months should be calculated."
    assert "SepsisLabel" not in test_df.columns, "Sepsis label should not be present in test data."