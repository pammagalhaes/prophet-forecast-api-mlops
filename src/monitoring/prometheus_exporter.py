from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

drift_metric = Gauge(
    "data_drift_detected",
    "Indica se drift foi detectado (1 = sim, 0 = não)"
)


def update_drift_metrics(report: dict):
    drift_found = int(report["metrics"][0]["result"]["drift_detected"])
    drift_metric.set(drift_found)


def prometheus_response():
    return generate_latest(), CONTENT_TYPE_LATEST

