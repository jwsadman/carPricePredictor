# app/model_utils.py
import mlflow.pyfunc
import pandas as pd
import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Set these appropriately
MLFLOW_TRACKING_URI = "http://ec2-13-220-203-71.compute-1.amazonaws.com:5000/"
MODEL_NAME = "price_predictor_model"
MODEL_STAGE = "Staging"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
    return root_dir

def load_model():
    """Load the MLflow model from the specified stage."""
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    return mlflow.pyfunc.load_model(model_uri)


def load_label_encoders() -> dict:
    """Load saved label encoders dictionary from pickle."""
    root_dir = get_root_directory()
    encoder_path = os.path.join(root_dir, 'carPricePredictor', 'label_encoders.pkl')

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {encoder_path}")
    
    try:
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        logger.debug("LabelEncoders loaded from %s", encoder_path)
        return encoders
    except Exception as e:
        logger.error("Failed to load label encoders: %s", e)
        raise



def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply saved label encoders to dataframe."""
    high_cardinality_columns = ['make', 'model', 'body']

    for col in high_cardinality_columns:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            if 'Unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'Unknown')
            df[col] = le.transform(df[col])
        else:
            raise ValueError(f"Missing encoder for column: {col}")
    
    return df


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess raw input dictionary into model-ready format."""
    df = pd.DataFrame([data])

    # Normalize string columns
    for col in ['make', 'model', 'body']:
        if col in df.columns:
            df[col] = df[col].str.lower().fillna('unknown')

    # Load and apply label encoders
    encoders = load_label_encoders()
    df = encode_features(df, encoders)
    
    return df
