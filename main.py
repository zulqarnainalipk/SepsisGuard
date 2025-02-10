import argparse
import logging
from src.data_loader import load_dataset
from src.preprocessor import SepsisPreprocessor
from src.model import SepsisModel
from src.train import train_pipeline
from src.predict import predict
import yaml

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sepsis_prediction.log'),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pediatric Sepsis Prediction System')
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the sepsis prediction model')
    train_parser.add_argument('--config', type=str, required=True,
                            help='Path to configuration YAML file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate sepsis predictions')
    predict_parser.add_argument('--input', type=str, required=True,
                              help='Path to input test data directory')
    predict_parser.add_argument('--output', type=str, required=True,
                              help='Path to save prediction results')
    predict_parser.add_argument('--config', type=str, required=True,
                              help='Path to configuration YAML file')

    return parser.parse_args()

def load_config(config_path: str) -> dict:
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise

def main():
    configure_logging()
    args = parse_arguments()
    config = load_config(args.config)

    if args.command == 'train':
        logging.info("Starting training pipeline")
        train_pipeline(config)
        logging.info("Training completed successfully")

    elif args.command == 'predict':
        logging.info("Starting prediction pipeline")
        predict(
            input_path=args.input,
            output_path=args.output,
            config=config
        )
        logging.info(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()