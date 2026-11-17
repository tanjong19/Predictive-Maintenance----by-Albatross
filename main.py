from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "predictive_maintenance_full.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="UpTime Predictive Maintenance API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df):,} rows")
print(f"Machines: {df['machine_id'].nunique()}")

# =========================================================
# LOAD MODELS
# =========================================================

anomaly_bundle = joblib.load(
    os.path.join(
        MODEL_DIR,
        "anomaly_model.joblib"
    )
)

condition_bundle = joblib.load(
    os.path.join(
        MODEL_DIR,
        "condition_model.joblib"
    )
)

rul_bundle = joblib.load(
    os.path.join(
        MODEL_DIR,
        "rul_model.joblib"
    )
)

anomaly_model = anomaly_bundle["model"]
condition_model = condition_bundle["model"]
rul_model = rul_bundle["model"]

print("Models loaded successfully")

# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "temperature",
    "vibration",
    "motor_current",
    "rpm",
    "operating_hours",
    "workload",

    "temperature_avg",
    "temperature_slope",
    "temperature_std",

    "vibration_avg",
    "vibration_slope",
    "vibration_std",

    "motor_current_avg",
    "motor_current_slope",
    "motor_current_std",

    "rpm_avg",
    "rpm_slope",
    "rpm_std",

    "type_code"
]

# =========================================================
# MACHINE TYPE ENCODING
# =========================================================

machine_types = sorted(
    df["machine_type"].dropna().unique()
)

TYPE_MAP = {
    machine_type: index
    for index, machine_type
    in enumerate(machine_types)
}

print("Machine type mapping:")
print(TYPE_MAP)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def calculate_slope(values):

    values = np.asarray(values)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    try:
        return float(
            np.polyfit(x, values, 1)[0]
        )
    except Exception:
        return 0.0


def build_features(machine_data):

    data = machine_data.copy()

    # Sort chronologically according to existing order.
    # Dataset timestamps are elapsed-time strings, so preserve
    # their existing order within each machine.
    data = data.reset_index(drop=True)

    window = 5

    # Current sensor values
    sensors = [
        "temperature",
        "vibration",
        "motor_current",
        "rpm"
    ]

    for sensor in sensors:

        data[f"{sensor}_avg"] = (
            data[sensor]
            .rolling(
                window,
                min_periods=1
            )
            .mean()
        )

        data[f"{sensor}_std"] = (
            data[sensor]
            .rolling(
                window,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

        data[f"{sensor}_slope"] = (
            data[sensor]
            .rolling(
                window,
                min_periods=2
            )
            .apply(
                calculate_slope,
                raw=True
            )
            .fillna(0)
        )

    data["type_code"] = (
        data["machine_type"]
        .map(TYPE_MAP)
        .fillna(0)
    )

    return data


# =========================================================
# PREDICTION
# =========================================================

def predict_machine(machine_data):

    features_data = build_features(
        machine_data
    )

    latest = features_data.iloc[-1]

    X = pd.DataFrame(
        [[latest[col] for col in FEATURES]],
        columns=FEATURES
    )

    # -------------------------------
    # ANOMALY
    # -------------------------------

    anomaly_prediction = anomaly_model.predict(X)[0]

    anomaly_decision = anomaly_model.decision_function(X)[0]

    # Convert decision function to simple
    # demo anomaly score.
    anomaly_score = float(
        np.clip(
            0.5 - anomaly_decision,
            0,
            1
        )
    )

    is_anomaly = bool(
        anomaly_prediction == -1
    )

    # -------------------------------
    # CONDITION
    # -------------------------------

    condition_prediction = (
        condition_model.predict(X)[0]
    )

    condition = str(
        condition_prediction
    )

    # -------------------------------
    # RUL
    # -------------------------------

    rul_prediction = float(
        rul_model.predict(X)[0]
    )

    rul_prediction = max(
        0,
        rul_prediction
    )

    # -------------------------------
    # RISK
    # -------------------------------

    # Prototype risk calculation.
    condition_lower = condition.lower()

    if condition_lower == "critical":
        condition_risk = 90

    elif condition_lower == "warning":
        condition_risk = 60

    else:
        condition_risk = 20

    anomaly_risk = anomaly_score * 100

    # RUL contribution
    if rul_prediction <= 24:
        rul_risk = 90

    elif rul_prediction <= 72:
        rul_risk = 65

    elif rul_prediction <= 168:
        rul_risk = 40

    else:
        rul_risk = 15

    risk_score = (
        condition_risk * 0.45
        + anomaly_risk * 0.30
        + rul_risk * 0.25
    )

    risk_score = round(
        min(100, max(0, risk_score))
    )

    # -------------------------------
    # RESULT
    # -------------------------------

    return {
        "condition": condition,
        "anomaly": is_anomaly,
        "anomaly_score": round(
            anomaly_score,
            3
        ),
        "rul_hours": round(
            rul_prediction,
            1
        ),
        "risk_score": risk_score
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "models": {
            "anomaly": "loaded",
            "condition": "loaded",
            "rul": "loaded"
        },
        "dataset_rows": len(df),
        "machines": int(
            df["machine_id"].nunique()
        )
    }


# =========================================================
# MACHINES
# =========================================================

@app.get("/api/machines")
def get_machines():

    result = []

    for machine_id in df["machine_id"].unique():

        machine_data = df[
            df["machine_id"] == machine_id
        ]

        latest = machine_data.iloc[-1]

        prediction = predict_machine(
            machine_data
        )

        result.append({
            "machine_id": machine_id,
            "machine_type": latest["machine_type"],
            **prediction
        })

    return result


# =========================================================
# MACHINE DETAILS
# =========================================================

@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str):

    machine_data = df[
        df["machine_id"] == machine_id
    ]

    if machine_data.empty:
        raise HTTPException(
            status_code=404,
            detail="Machine not found"
        )

    latest = machine_data.iloc[-1]

    prediction = predict_machine(
        machine_data
    )

    return {
        "machine_id": machine_id,
        "machine_type": latest["machine_type"],

        "temperature": float(
            latest["temperature"]
        ),

        "vibration": float(
            latest["vibration"]
        ),

        "motor_current": float(
            latest["motor_current"]
        ),

        "rpm": float(
            latest["rpm"]
        ),

        "workload": float(
            latest["workload"]
        ),

        "operating_hours": float(
            latest["operating_hours"]
        ),

        **prediction
    }


# =========================================================
# MACHINE HISTORY
# =========================================================

@app.get("/api/machines/{machine_id}/history")
def get_machine_history(
    machine_id: str
):

    machine_data = df[
        df["machine_id"] == machine_id
    ]

    if machine_data.empty:
        raise HTTPException(
            status_code=404,
            detail="Machine not found"
        )

    # Last 100 readings
    machine_data = machine_data.tail(100)

    history = []

    for _, row in machine_data.iterrows():

        history.append({
            "timestamp": str(
                row["timestamp"]
            ),
            "temperature": float(
                row["temperature"]
            ),
            "vibration": float(
                row["vibration"]
            ),
            "motor_current": float(
                row["motor_current"]
            ),
            "rpm": float(
                row["rpm"]
            ),
            "workload": float(
                row["workload"]
            )
        })

    return history


# =========================================================
# DASHBOARD
# =========================================================

@app.get("/api/dashboard")
def dashboard():

    machines = get_machines()

    normal = sum(
        1 for m in machines
        if m["condition"].lower()
        == "normal"
    )

    warning = sum(
        1 for m in machines
        if m["condition"].lower()
        == "warning"
    )

    critical = sum(
        1 for m in machines
        if m["condition"].lower()
        == "critical"
    )

    anomalies = sum(
        1 for m in machines
        if m["anomaly"]
    )

    return {
        "total_machines": len(machines),
        "normal": normal,
        "warning": warning,
        "critical": critical,
        "anomalies": anomalies
    }


# =========================================================
# ALERTS
# =========================================================

@app.get("/api/alerts")
def alerts():

    machines = get_machines()

    alerts = []

    for machine in machines:

        if (
            machine["condition"].lower()
            != "normal"
            or machine["anomaly"]
        ):

            alerts.append({
                "machine_id":
                    machine["machine_id"],

                "severity":
                    machine["condition"],

                "risk_score":
                    machine["risk_score"],

                "rul_hours":
                    machine["rul_hours"],

                "anomaly":
                    machine["anomaly"],

                "message":
                    "Machine requires maintenance attention."
            })

    return alerts