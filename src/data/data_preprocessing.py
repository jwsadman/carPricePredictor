import os
import re
import logging
import pandas as pd
from scipy.stats import zscore

# -------------------- Logging Configuration --------------------
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------   Data Preprocessing --------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess dataset."""
    try:
        df = df.copy()
        logger.debug("Starting data preprocessing")

        # Convert saledate to datetime
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce', utc=True)
        df['saleyear'] = df['saledate'].dt.year
        df['salemonth'] = df['saledate'].dt.month

        # Drop rows with missing make and model
        df.dropna(subset=['make', 'model'], how='all', inplace=True)

        # Fill missing numeric values with mean
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mean(), inplace=True)

        # Fill missing categorical values
        df['model'].fillna('Unknown', inplace=True)
        df['trim'].fillna('Unknown', inplace=True)
        df['body'].fillna('Unknown', inplace=True)
        df['transmission'].fillna('Unknown', inplace=True)

        if 'color' in df.columns:
            df['color'].fillna(df['color'].mode()[0], inplace=True)
        if 'interior' in df.columns:
            df['interior'].fillna(df['interior'].mode()[0], inplace=True)

        # Z-score outlier removal
        if 'odometer' in df.columns:
            df['odometer_zscore'] = zscore(df['odometer'])
        if 'sellingprice' in df.columns:
            df['sellingprice_zscore'] = zscore(df['sellingprice'])

        df = df[(df['odometer_zscore'].abs() <= 3) & (df['sellingprice_zscore'].abs() <= 3)]

        # Drop unnecessary columns
        drop_cols = ['vin', 'saledate', 'odometer_zscore', 'sellingprice_zscore',
                     'year', 'trim', 'transmission', 'state', 'color',
                     'interior', 'seller', 'mmr', 'salemonth']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # Normalize text columns
        df['make'] = df['make'].str.lower()
        df['model'] = df['model'].str.lower()
        df['body'] = df['body'].str.lower()

        logger.debug("data preprocessing completed")
        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise


# -------------------- Save Processed Data --------------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed train and test data."""
    try:
        interim_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_path, exist_ok=True)
        logger.debug(f"Saving processed data to {interim_path}")

        train_data.to_csv(os.path.join(interim_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_path, "test_processed.csv"), index=False)

        logger.debug("Processed data saved successfully")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


# -------------------- Main Execution --------------------
def main():
    try:
        logger.debug("Starting   dataset preprocessing pipeline...")

        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Raw data loaded")

        train_processed = preprocess_data(train_data)
        test_processed = preprocess_data(test_data)

        save_data(train_processed, test_processed, data_path='./data')
    except Exception as e:
        logger.error(f"Failed to complete preprocessing: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
