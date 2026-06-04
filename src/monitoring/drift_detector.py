import time

from src.monitoring.drift_report import generate_reference_and_current, run_drift_report

DRIFT_CACHE_TTL_SECONDS = 30
_cached_drift_result = None
_cached_drift_timestamp = 0.0

def check_drift():
    global _cached_drift_result, _cached_drift_timestamp

    now = time.time()
    if _cached_drift_result is not None and now - _cached_drift_timestamp < DRIFT_CACHE_TTL_SECONDS:
        return _cached_drift_result

    ref, cur = generate_reference_and_current()
    report = run_drift_report(ref, cur)

    result = report["metrics"][0]["result"]

    share_drifted = result.get("share_of_drifted_columns", 0)
    drift_detected = result.get("dataset_drift", False)

    _cached_drift_result = (share_drifted, drift_detected, report)
    _cached_drift_timestamp = now

    return _cached_drift_result
