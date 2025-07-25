import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import re

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

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

def load_data(data_url: str, **kwargs) -> pd.DataFrame:
    """Load data from a CSV file or Google Drive link with optional parameters."""
    try:
        # Detect and handle Google Drive shared links
        match = re.match(r"https://drive\.google\.com/file/d/([^/]+)/view", data_url)
        if match:
            file_id = match.group(1)
            data_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            logger.info("Converted Google Drive URL to direct download link: %s", data_url)

        df = pd.read_csv(data_url, **kwargs)
        logger.debug("Data loaded from %s", data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load data from the specified URL
        df = load_data(data_url='https://drive.google.com/file/d/1CcjJ23cUOp3IQaMMw7aH2it3Amzq6i25/view?usp=sharing',encoding="utf-8",quotechar='"',delimiter=',',on_bad_lines="skip",engine="python")

        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()