import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_encoders(encoder_path: str) -> dict:
    """Load all saved LabelEncoders as a dictionary."""
    try:
        with open(encoder_path, 'rb') as file:
            encoders = pickle.load(file)
        logger.debug('LabelEncoders loaded from %s', encoder_path)
        return encoders
    except Exception as e:
        logger.error('Error loading encoders from %s: %s', encoder_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def prepare_data(data: pd.DataFrame, encoders: dict, target_column: str = 'sellingprice'):
    high_cardinality_columns = ['make', 'model', 'body']

    for col in high_cardinality_columns:
        if col in data.columns and col in encoders:
            le = encoders[col]
            data[col] = data[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')  # Add 'Unknown' to classes if missing
            data[col] = le.transform(data[col])
        else:
            raise ValueError(f"Missing encoder for column: {col}")

    X_test = data.drop(columns=[target_column])  
    y_test = data[target_column]
    
    return X_test, y_test




def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return regression metrics."""
    try:
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.debug('Model evaluation completed')

        metrics = {
            "mse": mse,
            "mae": mae,
            "r2_score": r2
        }

        return metrics
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://ec2-13-220-203-71.compute-1.amazonaws.com:5000/")

    mlflow.set_experiment('dvc-pipeline-runs')
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '300'
    with mlflow.start_run() as run:
        try:
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Load model and Encoder
            model = load_model(os.path.join(root_dir, 'xgb_model.pkl'))
            encoders = load_encoders(os.path.join(root_dir, 'label_encoders.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test, y_test = prepare_data(test_data, encoders)

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test[:5])  # <--- Added for signature

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test[:5]))  # <--- Added for signature

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "xgb_model",
                signature=signature,  # <--- Added for signature
                input_example=input_example  # <--- Added input example
            )

            # Save model info
            # artifact_uri = mlflow.get_artifact_uri()
            model_path = "xgb_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Log the encoder as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'label_encoders.pkl'))

            # Evaluate model and get regression metrics
            metrics = evaluate_model(model, X_test, y_test)

            # Log regression metrics
            mlflow.log_metric("test_mse", metrics["mse"])
            mlflow.log_metric("test_mae", metrics["mae"])
            mlflow.log_metric("test_r2_score", metrics["r2_score"])


            # Add important tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("task", "Price Predictor")
            mlflow.set_tag("dataset", "Cars")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
