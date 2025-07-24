import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def encode_features_and_split(data: pd.DataFrame, target_column: str = 'sellingprice'):
    """
    Encodes high-cardinality categorical columns and splits features and target.

    Parameters:
    - data (pd.DataFrame): The input dataframe with categorical features.
    - target_column (str): The name of the target column.

    Returns:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    """
    le = LabelEncoder()
    high_cardinality_columns = ['make', 'model', 'body']  
    
    for col in high_cardinality_columns:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].fillna('Unknown'))

    X_train = data.drop(columns=[target_column])  
    y_train = data[target_column]
    
    return X_train, y_train


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> xgb.XGBRegressor:
    """Train a XGBoost model."""
    try:
        best_model = xgb.XGBRegressor(
            gamma=0.15,
            reg_alpha=0.99,  # L1 regularization
            reg_lambda=0.026,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('XGBoost model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during XGBoost model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply feature encoding on training data
        X_train, y_train = encode_features_and_split(train_data)

        # Train the XGBoost model using hyperparameters from params.yaml
        best_model = train_xgb(X_train, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'xgb_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature encoding and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()