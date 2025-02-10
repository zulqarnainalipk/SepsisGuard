from .data_loader import load_dataset
from .preprocessor import SepsisPreprocessor
from .model import SepsisModel
import yaml
import joblib

def train():
    # Load config
    with open('config/paths.yaml') as f:
        config = yaml.safe_load(f)
        
    # Load data
    train_data = load_dataset(config['train_data_path'], 'train')
    
    # Preprocess
    preprocessor = SepsisPreprocessor()
    train_df = preprocessor.preprocess(train_data, is_train=True)
    
    # Train model
    model = SepsisModel({
        'n_estimators': 500,
        'max_depth': 5,
        'random_state': 42
    })
    model.train(
        train_df.drop(['SepsisLabel', 'person_id', 'measurement_datetime'], axis=1),
        train_df['SepsisLabel']
    )
    
    # Save artifacts
    model.save(config['model_path'])
    joblib.dump(preprocessor.encoders, config['encoder_path'])
    joblib.dump(preprocessor.medians, config['median_path'])