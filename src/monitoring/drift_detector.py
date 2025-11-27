from src.monitoring.drift_report import generate_reference_and_current, run_drift_report

def check_drift():

    ref, cur = generate_reference_and_current()
    report = run_drift_report(ref, cur)

    result = report["metrics"][0]["result"]

    share_drifted = result.get("share_of_drifted_columns", 0)
    drift_detected = result.get("dataset_drift", False)

    return share_drifted, drift_detected, report
