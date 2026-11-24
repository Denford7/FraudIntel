from prometheus_client import Counter, Histogram


# Count incoming API requests
REQUEST_COUNT = Counter(
    "fraudintel_http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

# Measure API response time
REQUEST_LATENCY = Histogram(
    "fraudintel_http_request_duration_seconds",
    "HTTP request processing time in seconds",
    ["method", "endpoint"],
)

# Count transactions scored by the fraud model
PREDICTIONS_TOTAL = Counter(
    "fraudintel_predictions_total",
    "Total number of fraud predictions",
    ["prediction", "risk_level"],
)

# Count prediction failures
PREDICTION_ERRORS = Counter(
    "fraudintel_prediction_errors_total",
    "Total number of prediction errors",
)