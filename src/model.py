import joblib
from sklearn.ensemble import RandomForestClassifier

class SepsisModel:
    def __init__(self, params=None):
        self.model = RandomForestClassifier(**params) if params else None
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.model = joblib.load(path)
        return instance