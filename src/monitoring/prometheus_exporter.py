from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# Métrica que indica se drift foi detectado (1 sim, 0 não)
drift_metric = Gauge(
    "data_drift_detected",
    "Indica se drift foi detectado (1 = sim, 0 = não)"
)

# Agora recebe APENAS 1 ou 0
def update_drift_metrics(drift_detected: bool | int | float):
    drift_metric.set(int(drift_detected))

def prometheus_response():
    return generate_latest(), CONTENT_TYPE_LATEST