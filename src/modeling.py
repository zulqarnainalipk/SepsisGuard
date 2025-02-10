from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def train_random_forest(X, y):
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        verbose=1
    )
    rf_model.fit(X, y)
    return rf_model

def evaluate_model(y_true, y_pred, y_pred_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred_proba),
        "PR_AUC": average_precision_score(y_true, y_pred_proba)
    }