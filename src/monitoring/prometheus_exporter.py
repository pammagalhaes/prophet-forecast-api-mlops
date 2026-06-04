from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# Metric indicating whether drift was detected (1 = yes, 0 = no)
drift_metric = Gauge(
    "data_drift_detected",
    "Indicates whether drift was detected (1 = yes, 0 = no)"
)

# Now accepts only 1 or 0
def update_drift_metrics(drift_detected: bool | int | float):
    drift_metric.set(int(drift_detected))

def prometheus_response():
    return generate_latest(), CONTENT_TYPE_LATEST