import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data.load_data import load_raw
from src.data.preprocess import basic_clean, to_prophet_df


def generate_reference_and_current(cutoff_date="2015-06-01"):
    df = load_raw()
    df = basic_clean(df)
    df = to_prophet_df(df)

    ref = df[df["ds"] < cutoff_date].copy()
    cur = df[df["ds"] >= cutoff_date].copy()

    return ref, cur


def run_drift_report(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report.as_dict()
