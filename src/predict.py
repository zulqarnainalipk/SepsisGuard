# src/predict.py
import os
import joblib
import pandas as pd
from tqdm.auto import tqdm
from .data_loader import load_dataset
from .preprocessor import SepsisPreprocessor
import logging

logger = logging.getLogger(__name__)

def predict(input_data_path: str, output_path: str, model_path: str, encoder_path: str):
    """
    Generate sepsis predictions for new data
    Args:
        input_data_path: Path to directory containing test data files
        output_path: Where to save predictions CSV
        model_path: Path to saved model (joblib file)
        encoder_path: Path to saved encoders (joblib file)
    """
    try:
        # 1. Load artifacts
        logger.info("Loading trained artifacts...")
        model = joblib.load(model_path)
        preprocessor = SepsisPreprocessor()
        preprocessor.encoders = joblib.load(encoder_path)
        preprocessor.medians = joblib.load(median_path)  # NEW LINE

        # 2. Load and validate test data
        logger.info("Loading test data...")
        test_data = load_dataset(input_data_path, 'test')
        
        if test_data['sepsis_labels'] is None:
            raise ValueError("Missing sepsis labels file in test data")

        # 3. Preprocess test data
        logger.info("Preprocessing test data...")
        test_df = preprocessor.preprocess(test_data, is_train=False)
        
        # 4. Prepare features (match training structure)
        features_to_drop = ['person_id', 'measurement_datetime']
        if 'SepsisLabel' in test_df.columns:
            test_features = test_df.drop(features_to_drop + ['SepsisLabel'], axis=1, errors='ignore')
        else:
            test_features = test_df.drop(features_to_drop, axis=1, errors='ignore')

        # 5. Generate predictions
        logger.info("Generating predictions...")
        predictions = model.predict_proba(test_features)[:, 1]  # Get probability scores
        
        # 6. Format output
        logger.info("Formatting output...")
        test_df['person_id_datetime'] = (
            test_df['person_id'].astype(str) + '_' + 
            test_df['measurement_datetime'].astype(str)
        )
        result_df = pd.DataFrame({
            'person_id_datetime': test_df['person_id_datetime'],
            'SepsisLabel': predictions
        })
        
        # 7. Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for predictions CSV')
    parser.add_argument('--model', type=str, default='models/rf_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--encoders', type=str, default='models/encoders.pkl',
                       help='Path to saved encoders')
    args = parser.parse_args()
    
    predict(
        input_data_path=args.input,
        output_path=args.output,
        model_path=args.model,
        encoder_path=args.encoders
    )