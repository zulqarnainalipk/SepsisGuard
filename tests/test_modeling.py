import os
import pandas as pd
from src.modeling import train_random_forest, evaluate_model

def test_model_training_and_submission():
    # Mock training features and labels
    X_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "feature_2": [5, 6, 7, 8]
    })
    y_train = pd.Series([0, 0, 1, 1])

    # Train the Random Forest model using the function from src/modeling.py
    rf_model = train_random_forest(X_train, y_train)

    # Mock test features
    X_test = pd.DataFrame({
        "feature_1": [2, 3],
        "feature_2": [6, 7]
    })

    # Generate predictions for the test set
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Evaluate metrics on the training set
    y_train_pred_proba = rf_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    metrics = evaluate_model(y_train, y_train_pred, y_train_pred_proba)

    # Assertions for training metrics
    assert "Accuracy" in metrics, "Metrics should include Accuracy."
    assert "F1_Score" in metrics, "Metrics should include F1 Score."
    assert "AUC" in metrics, "Metrics should include AUC."
    assert "PR_AUC" in metrics, "Metrics should include PR AUC."
    assert metrics["Accuracy"] > 0, "Accuracy should be greater than 0."
    assert metrics["F1_Score"] > 0, "F1 Score should be greater than 0."

    # Create a mock submission DataFrame
    test_df = pd.DataFrame({
        "person_id_datetime": ["1_2023-01-01 10:00:00", "2_2023-01-01 11:00:00"]
    })
    test_df["SepsisLabel"] = y_pred_proba

    # Save the submission file
    submission_path = "mock_submission.csv"
    test_df.to_csv(submission_path, index=False)

    # Validate the submission file
    submission_df = pd.read_csv(submission_path)
    assert "person_id_datetime" in submission_df.columns, "Submission file should contain 'person_id_datetime'."
    assert "SepsisLabel" in submission_df.columns, "Submission file should contain 'SepsisLabel'."
    assert len(submission_df) == len(X_test), "Submission file should have the same number of rows as test data."

    print("Model training and submission test passed!")