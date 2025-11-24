import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DataDriftTable


def generate_reference_data():
    return pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=30, freq="D"),
        "y": 100 + np.random.normal(0, 5, 30)
    })


def generate_current_data():
    return pd.DataFrame({
        "ds": pd.date_range("2025-02-01", periods=30, freq="D"),
        "y": 110 + np.random.normal(0, 5, 30)  
    })


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    report = Report(metrics=[DataDriftTable()])

    report.run(
        reference_data=reference_df,
        current_data=current_df
    )

    return report.as_dict()

