import time

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Gauge, generate_latest
from time_rcd import TimeRCDDetector


app = FastAPI()

MIMIR_URL = "http://mimir-gateway.monitoring.svc.cluster.local"
NAMESPACE = "banking-app"

SERVICES = [
    "accounts-service",
    "api-gateway",
    "ai-service",
    "fraud-service",
    "frontend",
    "notification-service",
    "transactions-service",
    "user-service",
]

# Initial per-service thresholds based on current observed normal scores.
# These are demo/baseline thresholds and should later be calibrated
# from a longer normal-history window.
THRESHOLDS = {
    "accounts-service": 0.040,
    "api-gateway": 0.025,
    "ai-service": 0.005,
    "fraud-service": 0.005,
    "frontend": 0.040,
    "notification-service": 0.030,
    "transactions-service": 0.035,
    "user-service": 0.035,
}


detector = TimeRCDDetector.from_pretrained(
    variant="multi",
    device="cpu"
)


anomaly = Gauge(
    "predictops_anomaly_score",
    "Latest anomaly score",
    ["target_service"]
)

threshold_metric = Gauge(
    "predictops_anomaly_threshold",
    "Anomaly threshold for each service",
    ["target_service"]
)

anomaly_state = Gauge(
    "predictops_is_anomaly",
    "Whether the service is currently anomalous",
    ["target_service"]
)


def query_mimir(promql):
    end = int(time.time())
    start = end - 600

    response = requests.get(
        f"{MIMIR_URL}/prometheus/api/v1/query_range",
        params={
            "query": promql,
            "start": start,
            "end": end,
            "step": 30,
        },
        timeout=10,
    )

    response.raise_for_status()
    return response.json()


def score_service(service):
    cpu_query = f"""
    sum(
      rate(container_cpu_usage_seconds_total{{
        namespace="{NAMESPACE}",
        container="{service}"
      }}[5m])
    )
    """

    memory_query = f"""
    sum(
      container_memory_working_set_bytes{{
        namespace="{NAMESPACE}",
        container="{service}"
      }}
    )
    """

    cpu_data = query_mimir(cpu_query)
    memory_data = query_mimir(memory_query)

    cpu_results = cpu_data["data"]["result"]
    memory_results = memory_data["data"]["result"]

    if not cpu_results or not memory_results:
        return {
            "service": service,
            "status": "no_data",
        }

    cpu_values = [
        float(value[1])
        for value in cpu_results[0]["values"]
    ]

    memory_values = [
        float(value[1])
        for value in memory_results[0]["values"]
    ]

    n = min(
        len(cpu_values),
        len(memory_values),
    )

    if n < 5:
        return {
            "service": service,
            "status": "insufficient_data",
            "points": n,
        }

    window = np.column_stack([
        cpu_values[-n:],
        memory_values[-n:],
    ])

    scores = detector.predict(window)

    latest_score = round(
        float(np.asarray(scores).reshape(-1)[-1]),
        5
    )

    service_threshold = round(THRESHOLDS[service], 5)
    is_anomaly = latest_score > service_threshold

    anomaly.labels(
        target_service=service
    ).set(latest_score)

    threshold_metric.labels(
        target_service=service
    ).set(service_threshold)

    anomaly_state.labels(
        target_service=service
    ).set(1 if is_anomaly else 0)

    return {
        "service": service,
        "anomaly_score": latest_score,
        "threshold": service_threshold,
        "is_anomaly": is_anomaly,
        "points": n,
    }


@app.post("/score")
def score_all_services():
    results = []

    for service in SERVICES:
        try:
            result = score_service(service)
            results.append(result)

        except Exception as exc:
            results.append({
                "service": service,
                "status": "error",
                "error": str(exc),
            })

    return {
        "namespace": NAMESPACE,
        "results": results,
    }


@app.post("/score/{service}")
def score_single_service(service: str):
    if service not in SERVICES:
        return {
            "error": "unknown service",
            "allowed_services": SERVICES,
        }

    try:
        return score_service(service)

    except Exception as exc:
        return {
            "service": service,
            "status": "error",
            "error": str(exc),
        }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "namespace": NAMESPACE,
        "services": SERVICES,
    }