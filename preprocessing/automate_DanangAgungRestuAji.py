"""
Automated Demand Preprocessing (preprocessing-only)
Converted from Eksperimen_DanangAgungRestuAji.ipynb

Purpose: provide a reusable, modular preprocessing script that matches
the notebook steps for demand forecasting (NOT modelling).

Outputs produced:
- cleaned_amazon_sales.csv
- daily_demand_forecasting.csv
- daily_demand_by_state.csv (optional)
- daily_demand_by_sku.csv (optional)

This module intentionally does NOT perform:
- scaling
- train/val/test splitting
- label encoding of the target
- saving/loading modelling artifacts

Those belong to a separate modelling pipeline.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG (match notebook filenames)
# =============================================================================
INPUT_FILE = "../amazon_sales_raw/amazon_sale_raw.csv"
CLEAN_OUTPUT = "cleaned_amazon_sales.csv"
TS_OUTPUT = "daily_demand_forecasting.csv"
STATE_OUTPUT = "daily_demand_by_state.csv"
SKU_OUTPUT = "daily_demand_by_sku.csv"

VALID_STATUS = {"Shipped", "Shipped - Delivered to Buyer"}

TEXT_COLUMNS = [
    "Status",
    "Fulfilment",
    "Sales Channel ",
    "ship-service-level",
    "Style",
    "SKU",
    "Category",
    "Size",
    "ASIN",
    "Courier Status",
    "ship-city",
    "ship-state",
    "ship-country",
    "fulfilled-by",
]

SHIPPING_COLUMNS = ["ship-city", "ship-state", "ship-postal-code", "ship-country"]


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]


class AmazonDemandPreprocessor:
    """Preprocessor that implements the notebook's cleaning and feature engineering
    for demand forecasting (per category / per state / per SKU).
    """

    def __init__(self, input_path: Optional[str] = None):
        self.input_path = input_path or INPUT_FILE
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        print_section("LOAD DATA")
        path = Path(self.input_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
        self.df = pd.read_csv(path)
        print(f"Dataset Loaded: {path}")
        print(f"Initial Shape: {self.df.shape}")
        return self.df

    def remove_duplicates(self):
        print_section("REMOVE DUPLICATES")
        if self.df is None:
            raise RuntimeError("Data not loaded")
        if "Order ID" in self.df.columns:
            before = len(self.df)
            self.df = self.df.drop_duplicates(subset=["Order ID"])
            print(f"Duplicates Removed Using Order ID: {before - len(self.df)}")
        else:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            print(f"Duplicates Removed: {before - len(self.df)}")
        print(f"Shape After Deduplication: {self.df.shape}")

    def drop_unused_columns(self):
        print_section("DROP UNUSED COLUMNS")
        drop_cols = [col for col in self.df.columns if "unnamed" in col.lower()]
        if drop_cols:
            self.df = self.df.drop(columns=drop_cols)
            print(f"Dropped Columns: {drop_cols}")
        print(f"Remaining Columns: {len(self.df.columns)}")

    def clean_dates(self):
        print_section("DATE CLEANING")
        if "Date" not in self.df.columns:
            raise KeyError("Column 'Date' not found.")
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%m-%d-%y", errors="coerce")
        invalid_dates = self.df["Date"].isnull().sum()
        print(f"Invalid Dates Removed: {invalid_dates}")
        self.df = self.df.dropna(subset=["Date"])

    def numeric_cleaning(self):
        print_section("NUMERIC CLEANING")
        numeric_cols = ["Amount", "Qty"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        print("\nNumeric Null Counts:")
        print(self.df[[c for c in numeric_cols if c in self.df.columns]].isnull().sum())

    def status_filtering(self):
        print_section("STATUS FILTERING")
        if "Status" in self.df.columns:
            print("\nStatus Distribution Before:")
            print(self.df["Status"].value_counts(dropna=False))
            before = len(self.df)
            self.df = self.df[self.df["Status"].isin(VALID_STATUS)]
            after = len(self.df)
            print(f"\nRows Removed By Status Filter: {before - after}")
            print("\nStatus Distribution After:")
            print(self.df["Status"].value_counts(dropna=False))

    def remove_invalid_values(self):
        print_section("REMOVE INVALID VALUES")
        if "Qty" in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df["Qty"] > 0]
            print(f"Removed Invalid Qty Rows: {before - len(self.df)}")
        if "Amount" in self.df.columns:
            before = len(self.df)
            self.df = self.df.dropna(subset=["Amount"])
            self.df = self.df[self.df["Amount"] > 0]
            print(f"Removed Invalid Amount Rows: {before - len(self.df)}")

    def remove_outliers(self):
        print_section("REMOVE OUTLIERS")
        if "Amount" in self.df.columns:
            before = len(self.df)
            self.df = remove_outliers_iqr(self.df, "Amount")
            print(f"Outliers Removed: {before - len(self.df)}")

    def shipping_cleaning(self):
        print_section("SHIPPING DATA CLEANING")
        existing_shipping_cols = [col for col in SHIPPING_COLUMNS if col in self.df.columns]
        if existing_shipping_cols:
            before = len(self.df)
            self.df = self.df.dropna(subset=existing_shipping_cols)
            print(f"Rows Removed Due Missing Shipping Data: {before - len(self.df)}")

    def promotion_and_fulfillment(self):
        print_section("PROMOTION & FULFILLMENT CLEANING")
        if "promotion-ids" in self.df.columns:
            self.df["has_promo"] = self.df["promotion-ids"].notnull().astype(int)
            self.df["promotion-ids"] = self.df["promotion-ids"].fillna("No Promotion")
        else:
            self.df["has_promo"] = 0
            self.df["promotion-ids"] = "No Promotion"

        if "fulfilled-by" in self.df.columns:
            self.df["fulfilled-by"] = self.df["fulfilled-by"].fillna("Merchant")
        if "Courier Status" in self.df.columns:
            self.df["Courier Status"] = self.df["Courier Status"].fillna("Unknown")

    def currency_cleaning(self):
        print_section("CURRENCY CLEANING")
        if "currency" in self.df.columns:
            print("\nCurrency Distribution:")
            print(self.df["currency"].value_counts(dropna=False))
            self.df["currency"] = self.df["currency"].fillna("INR")
            if self.df["currency"].nunique() > 1:
                print("\n[WARNING] Multiple currencies detected. Filtering to INR rows.")
                before = len(self.df)
                self.df = self.df[self.df["currency"] == "INR"]
                print(f"Removed Non-INR Rows: {before - len(self.df)}")

    def text_cleaning(self):
        print_section("TEXT CLEANING")
        for col in TEXT_COLUMNS:
            if col in self.df.columns:
                self.df[col] = safe_strip(self.df[col])
        print("Text Cleaning Completed.")

    def data_type_fixing(self):
        print_section("DATA TYPE FIXING")
        if "Qty" in self.df.columns:
            # safe cast: fillna then astype int
            self.df["Qty"] = pd.to_numeric(self.df["Qty"], errors="coerce").fillna(0).astype(int)
        if "ship-postal-code" in self.df.columns:
            self.df["ship-postal-code"] = self.df["ship-postal-code"].astype(str).str.strip()
        if "B2B" in self.df.columns:
            self.df["B2B"] = self.df["B2B"].astype(bool).astype(int)
        print(self.df.dtypes)

    def sort_and_reset(self):
        print_section("SORTING")
        self.df = self.df.sort_values(by="Date")
        self.df = self.df.reset_index(drop=True)
        print("Data Sorted By Date.")

    def demand_aggregation(self):
        print_section("DEMAND AGGREGATION")
        if "Qty" not in self.df.columns:
            raise KeyError("Column 'Qty' not found for demand aggregation.")
        required_cols = {"Date", "Category"}
        if not required_cols.issubset(self.df.columns):
            raise KeyError("Columns 'Date' and 'Category' are required for demand per category.")

        demand_category = (
            self.df.groupby(["Date", "Category"]) ["Qty"]
            .sum()
            .reset_index()
        )
        demand_category = demand_category.rename(columns={"Qty": "Daily_Demand"})

        demand_state = None
        if "ship-state" in self.df.columns:
            demand_state = (
                self.df.groupby(["Date", "ship-state"]) ["Qty"]
                .sum()
                .reset_index()
            )
            demand_state = demand_state.rename(columns={"Qty": "Daily_Demand"})

        demand_sku = None
        if "SKU" in self.df.columns:
            demand_sku = (
                self.df.groupby(["Date", "SKU"]) ["Qty"]
                .sum()
                .reset_index()
            )
            demand_sku = demand_sku.rename(columns={"Qty": "Daily_Demand"})

        return demand_category, demand_state, demand_sku

    def create_time_features(self, category_ts: pd.DataFrame) -> pd.DataFrame:
        category_ts["day"] = category_ts["Date"].dt.day
        category_ts["month"] = category_ts["Date"].dt.month
        category_ts["year"] = category_ts["Date"].dt.year
        category_ts["weekday"] = category_ts["Date"].dt.weekday
        category_ts["weekofyear"] = (
            category_ts["Date"].dt.isocalendar().week.astype(int)
        )
        category_ts["quarter"] = category_ts["Date"].dt.quarter
        category_ts["is_weekend"] = (category_ts["weekday"] >= 5).astype(int)
        category_ts["month_start"] = category_ts["Date"].dt.is_month_start.astype(int)
        category_ts["month_end"] = category_ts["Date"].dt.is_month_end.astype(int)
        return category_ts

    def create_lag_features(self, category_ts: pd.DataFrame) -> pd.DataFrame:
        category_ts = category_ts.sort_values(["Category", "Date"])
        category_ts["lag_1"] = category_ts.groupby("Category")["Daily_Demand"].shift(1)
        category_ts["lag_7"] = category_ts.groupby("Category")["Daily_Demand"].shift(7)
        category_ts["lag_14"] = category_ts.groupby("Category")["Daily_Demand"].shift(14)
        category_ts["lag_30"] = category_ts.groupby("Category")["Daily_Demand"].shift(30)
        return category_ts

    def create_rolling_features(self, category_ts: pd.DataFrame) -> pd.DataFrame:
        category_ts["rolling_mean_7"] = (
            category_ts.groupby("Category")["Daily_Demand"].shift(1).rolling(window=7).mean()
        )
        category_ts["rolling_std_7"] = (
            category_ts.groupby("Category")["Daily_Demand"].shift(1).rolling(window=7).std()
        )
        category_ts["rolling_mean_30"] = (
            category_ts.groupby("Category")["Daily_Demand"].shift(1).rolling(window=30).mean()
        )
        category_ts["rolling_std_30"] = (
            category_ts.groupby("Category")["Daily_Demand"].shift(1).rolling(window=30).std()
        )
        return category_ts

    def drop_nulls_after_fe(self, category_ts: pd.DataFrame) -> pd.DataFrame:
        print_section("REMOVE NULLS AFTER FEATURE ENGINEERING")
        before = len(category_ts)
        category_ts = category_ts.dropna()
        after = len(category_ts)
        print(f"Rows Removed: {before - after}")
        print(f"Final Time Series Shape: {category_ts.shape}")
        return category_ts

    def save_outputs(self, cleaned: pd.DataFrame, category_ts: pd.DataFrame, demand_state: Optional[pd.DataFrame], demand_sku: Optional[pd.DataFrame], output_dir: str = '.'):
        print_section("SAVE OUTPUT")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cleaned.to_csv(out_dir / CLEAN_OUTPUT, index=False)
        category_ts.to_csv(out_dir / TS_OUTPUT, index=False)
        if demand_state is not None:
            demand_state.to_csv(out_dir / STATE_OUTPUT, index=False)
        if demand_sku is not None:
            demand_sku.to_csv(out_dir / SKU_OUTPUT, index=False)

        print(f"Saved Clean Dataset → {out_dir / CLEAN_OUTPUT}")
        print(f"Saved Demand (Category) Dataset → {out_dir / TS_OUTPUT}")
        if demand_state is not None:
            print(f"Saved Demand (State) Dataset → {out_dir / STATE_OUTPUT}")
        if demand_sku is not None:
            print(f"Saved Demand (SKU) Dataset → {out_dir / SKU_OUTPUT}")

    def run_pipeline(self, output_dir: str = '.') -> None:
        # Execute steps mirroring the notebook
        self.load_data()
        self.remove_duplicates()
        self.drop_unused_columns()
        self.clean_dates()
        self.numeric_cleaning()
        self.status_filtering()
        self.remove_invalid_values()
        self.remove_outliers()
        self.shipping_cleaning()
        self.promotion_and_fulfillment()
        self.currency_cleaning()
        self.text_cleaning()
        self.data_type_fixing()
        self.sort_and_reset()

        print_section("FINAL CLEANED DATA OVERVIEW")
        print(f"\nFinal Shape: {self.df.shape}")
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nData Types:")
        print(self.df.dtypes)

        # Aggregation and feature engineering
        demand_category, demand_state, demand_sku = self.demand_aggregation()

        print_section("TIME FEATURE ENGINEERING")
        category_ts = demand_category.copy()
        category_ts = self.create_time_features(category_ts)

        print_section("LAG FEATURE ENGINEERING")
        category_ts = self.create_lag_features(category_ts)

        print_section("ROLLING FEATURE ENGINEERING")
        category_ts = self.create_rolling_features(category_ts)

        category_ts = self.drop_nulls_after_fe(category_ts)

        print_section("FINAL TIME SERIES DATASET")
        print(category_ts.head())

        # Save outputs
        self.save_outputs(self.df, category_ts, demand_state, demand_sku, output_dir=output_dir)


if __name__ == "__main__":
    pre = AmazonDemandPreprocessor()
    pre.run_pipeline(output_dir='.')
