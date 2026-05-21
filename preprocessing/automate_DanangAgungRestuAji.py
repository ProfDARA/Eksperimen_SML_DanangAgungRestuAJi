"""
Automated Data Preprocessing Module for Amazon Sale Report
Author: Danang Agung Restu Aji
Purpose: Automate preprocessing steps from Eksperimen_DanangAgungRestuAji.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import json
from typing import Tuple, Dict, Any
import argparse


FEATURE_COLUMNS = [
    'lag_1',
    'lag_7',
    'lag_14',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'rolling_mean_30',
    'rolling_std_30',
    'day',
    'month',
    'year',
    'weekday',
    'weekofyear',
    'quarter',
    'is_weekend',
    'month_start',
    'month_end'
]

class AmazonSalePreprocessor:
    """
    Automated preprocessor for Amazon Sale Report dataset
    Performs all preprocessing steps: cleaning, encoding, scaling, and splitting
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """
        Initialize preprocessor
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set from training data
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.cleaning_report = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape}")
        return df
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with unnecessary columns removed
        """
        cols_to_drop = ['index', 'Unnamed: 22', 'Courier Status', 'promotion-ids']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        print(f"After dropping columns: {df.shape}")
        return df
    
    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        print(f"Removed {removed} duplicate rows. New shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Fill numeric missing values
        if 'ship-postal-code' in df.columns:
            df['ship-postal-code'] = df['ship-postal-code'].fillna(df['ship-postal-code'].median())
        
        if 'B2B' in df.columns:
            df['B2B'] = df['B2B'].fillna(False)
        
        if 'fulfilled-by' in df.columns:
            df['fulfilled-by'] = df['fulfilled-by'].fillna(df['fulfilled-by'].mode()[0])
        
        # Drop rows with critical missing values
        df = df.dropna(subset=['Amount', 'Status', 'Qty'])
        
        print(f"After handling missing values: {df.shape}")
        return df
    
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from Date column
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new date features
        """
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y', errors='coerce')
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df.drop(columns=['Date'], inplace=True)
            print("Date features extracted: Year, Month, DayOfWeek")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training, False for new data)
            
        Returns:
            DataFrame with encoded features
        """
        categorical_features = [
            'Status', 'Fulfilment', 'Sales Channel', 'ship-service-level',
            'Category', 'Size', 'ship-state', 'ship-country', 'fulfilled-by'
        ]
        
        for col in categorical_features:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    else:
                        raise ValueError(f"Encoder for {col} not fitted yet")
        
        # Drop original categorical columns
        df = df.drop(columns=[col for col in categorical_features if col in df.columns])
        print(f"Categorical features encoded: {len(categorical_features)} features")
        
        return df
    
    def scale_numeric_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numeric features
        
        Args:
            X: Input features DataFrame
            fit: Whether to fit scaler (True for training, False for new data)
            
        Returns:
            DataFrame with scaled features
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet")
            X_scaled = X.copy()
            X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        self.feature_names = X_scaled.columns.tolist()
        print(f"Numeric features scaled: {len(numeric_cols)} features")
        
        return X_scaled
    
    def preprocess(self, filepath: str, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to raw data CSV
            fit: Whether to fit preprocessors (True for training, False for new data)
            
        Returns:
            Tuple of (features, target)
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING PIPELINE STARTED")
        print("=" * 80 + "\n")
        
        # Load and clean data
        df = self.load_data(filepath)
        initial_rows = len(df)
        df = self.drop_unnecessary_columns(df)
        df = self.handle_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.extract_date_features(df)
        df = self.encode_categorical_features(df, fit=fit)
        
        # Separate features and target
        X = df.drop(columns=['Status_encoded']) if 'Status_encoded' in df.columns else df
        y = df['Status_encoded'] if 'Status_encoded' in df.columns else None

        # NOTE: Scaling must be done after train/test split to avoid data leakage.
        # The preprocessor will not fit or apply the scaler here. Modeling pipeline
        # should perform scaling and then save the fitted scaler for inference.
        
        # Dataset size check (rolling windows and lags reduce usable rows)
        if len(X) < 100:
            raise ValueError("Dataset terlalu kecil setelah preprocessing")

        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETED")
        print(f"Features shape: {X.shape}")
        if y is not None:
            print(f"Target shape: {y.shape}")
        print("=" * 80 + "\n")

        # Feature validation (ensure downstream modeling sees the same features)
        # Only enforce validation when at least one of the expected time-series
        # feature columns is present (i.e. time-series feature engineering ran).
        if 'FEATURE_COLUMNS' in globals():
            expected = set(FEATURE_COLUMNS)
            present = set(X.columns)
            # If none of the expected features are present, assume this is a
            # clean-only run and skip strict validation.
            if expected & present:
                missing_features = [col for col in FEATURE_COLUMNS if col not in X.columns]
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")

        # Prepare cleaning report
        self.cleaning_report = {
            "initial_rows": initial_rows,
            "final_rows": len(X),
            "removed_rows": initial_rows - len(X),
            "features_created": len(X.columns)
        }
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Decide whether stratification is safe: each class must have at least 2 samples
        stratify_first = None
        if y is not None:
            try:
                min_count = y.value_counts().min()
                if min_count >= 2:
                    stratify_first = y
                else:
                    print(f"Warning: stratify disabled because least populated class has {min_count} samples")
            except Exception:
                stratify_first = None

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_first
        )

        # For second split (train/val) decide stratify based on y_temp
        stratify_second = None
        if y_temp is not None:
            try:
                min_count_temp = y_temp.value_counts().min()
                if min_count_temp >= 2:
                    stratify_second = y_temp
                else:
                    print(f"Warning: validation stratify disabled because least populated class in temp split has {min_count_temp} samples")
            except Exception:
                stratify_second = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state, stratify=stratify_second
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def save_artifacts(self, directory: str = 'preprocessing_artifacts') -> None:
        """
        Save preprocessor artifacts (scaler and encoders)
        
        Args:
            directory: Directory to save artifacts
        """
        os.makedirs(directory, exist_ok=True)
        # Save scaler only if it was fitted (modeling pipeline responsibility)
        if self.scaler is not None:
            with open(f'{directory}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

        with open(f'{directory}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        with open(f'{directory}/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)

        # Save cleaning report if present
        if self.cleaning_report is not None:
            with open(os.path.join(directory, 'cleaning_report.json'), 'w') as f:
                json.dump(self.cleaning_report, f, indent=4)

        print(f"Artifacts saved to {directory}/")
    
    def load_artifacts(self, directory: str = 'preprocessing_artifacts') -> None:
        """
        Load preprocessor artifacts
        
        Args:
            directory: Directory containing artifacts
        """
        with open(f'{directory}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{directory}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(f'{directory}/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"Artifacts loaded from {directory}/")


def main_preprocessing_pipeline(raw_data_path: str, output_dir: str = 'preprocessing_artifacts') -> Dict[str, Any]:
    """
    Main function to run complete preprocessing pipeline
    
    Args:
        raw_data_path: Path to raw CSV data
        output_dir: Directory to save preprocessed data and artifacts
        
    Returns:
        Dictionary with preprocessed datasets and preprocessor object
    """
    # Initialize preprocessor
    preprocessor = AmazonSalePreprocessor(test_size=0.2, val_size=0.2, random_state=42)
    
    # Run preprocessing
    X, y = preprocessor.preprocess(raw_data_path, fit=True)
    
    # Split data
    data_splits = preprocessor.split_data(X, y)
    
    # Save artifacts
    preprocessor.save_artifacts(output_dir)
    
    # Save preprocessed datasets
    os.makedirs(output_dir, exist_ok=True)
    for split_name, (X_split, y_split) in data_splits.items():
        X_split.to_csv(f'{output_dir}/X_{split_name}.csv', index=False)
        y_split.to_csv(f'{output_dir}/y_{split_name}.csv', index=False)
    
    # Save cleaned time series dataset for demand forecasting (cleaning only, no scaling)
    cleaned_path = os.path.join(output_dir, 'daily_demand_forecasting.csv')
    X.to_csv(cleaned_path, index=False)

    # Save feature metadata for inference
    feature_metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "Category_Demand"
    }
    with open(os.path.join(output_dir, 'feature_metadata.pkl'), 'wb') as f:
        pickle.dump(feature_metadata, f)

    # Also copy important artifacts to the repository `preprocessing/` folder for easy access
    try:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        preprocessing_dir = os.path.join(repo_root, 'preprocessing')
        os.makedirs(preprocessing_dir, exist_ok=True)

        # copy cleaned csv
        import shutil
        shutil.copy(cleaned_path, os.path.join(preprocessing_dir, 'daily_demand_forecasting.csv'))

        # copy metadata and cleaning report if they exist in output_dir
        shutil.copy(os.path.join(output_dir, 'feature_metadata.pkl'), os.path.join(preprocessing_dir, 'feature_metadata.pkl'))
        if preprocessor.cleaning_report is not None:
            shutil.copy(os.path.join(output_dir, 'cleaning_report.json'), os.path.join(preprocessing_dir, 'cleaning_report.json'))
    except Exception:
        # non-fatal; copying is convenience only
        pass
    
    print(f"\nAll preprocessed data saved to {output_dir}/")
    
    return {
        'preprocessor': preprocessor,
        'data_splits': data_splits,
        'X': X,
        'y': y
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('-r', '--raw', default=None,
                        help='Path to raw CSV file (absolute or repo-relative). If omitted, script will try to locate the file relative to this script.')
    parser.add_argument('-o', '--output', default='preprocessing_artifacts',
                        help='Output directory to save artifacts')
    args = parser.parse_args()

    # Resolve raw data path: prefer provided arg, otherwise derive path relative to this script
    if args.raw:
        raw_data_path = args.raw
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # expected location: ../amazon_sales_raw/amazon_sale_raw.csv relative to preprocessing script
        raw_data_path = os.path.normpath(os.path.join(script_dir, '..', 'amazon_sales_raw', 'amazon_sale_raw.csv'))

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at resolved path: {raw_data_path}")

    result = main_preprocessing_pipeline(raw_data_path, output_dir=args.output)
    print('\nPreprocessing complete! Ready for model training.')
