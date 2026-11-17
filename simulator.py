import os
import time
import threading
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

ANOMALY_PATH = os.path.join(MODEL_DIR, "anomaly_model.joblib")
CONDITION_PATH = os.path.join(MODEL_DIR, "condition_model.joblib")
RUL_PATH = os.path.join(MODEL_DIR, "rul_model.joblib")


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

anomaly_bundle = joblib.load(ANOMALY_PATH)
condition_bundle = joblib.load(CONDITION_PATH)
rul_bundle = joblib.load(RUL_PATH)

anomaly_model = anomaly_bundle["model"]
condition_model = condition_bundle["model"]
rul_model = rul_bundle["model"]

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

    "type_code",
]


print("Models loaded successfully")


# ============================================================
# MACHINE SIMULATION
# ============================================================

machines = {

    "MILL-01": {
        "machine_type": "CNC Mill",
        "scenario": "normal",
        "temperature": 50,
        "vibration": 1.4,
        "motor_current": 7.5,
        "rpm": 6500,
        "workload": 52,
        "operating_hours": 3480,
    },

    "LATHE-01": {
        "machine_type": "CNC Lathe",
        "scenario": "degrading",
        "temperature": 55,
        "vibration": 1.8,
        "motor_current": 8.0,
        "rpm": 5900,
        "workload": 65,
        "operating_hours": 5200,
    },

    "PRINTER-01": {
        "machine_type": "FDM Printer",
        "scenario": "critical",
        "temperature": 70,
        "vibration": 3.2,
        "motor_current": 11.0,
        "rpm": 4300,
        "workload": 85,
        "operating_hours": 6800,
    },

}


# Same mapping used by the trained models
TYPE_CODE = {
    "CNC Lathe": 0,
    "CNC Mill": 1,
    "FDM Printer": 2,
}


# Store recent sensor readings
history = {
    machine_id: []
    for machine_id in machines
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_slope(values):

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    try:
        return float(np.polyfit(x, values, 1)[0])
    except Exception:
        return 0.0


def create_features(machine_id):

    machine = machines[machine_id]

    rows = history[machine_id]

    if len(rows) == 0:
        rows = [{
            "temperature": machine["temperature"],
            "vibration": machine["vibration"],
            "motor_current": machine["motor_current"],
            "rpm": machine["rpm"],
        }]

    df = pd.DataFrame(rows)

    window = df.tail(5)

    features = {}

    # Raw values
    features["temperature"] = float(machine["temperature"])
    features["vibration"] = float(machine["vibration"])
    features["motor_current"] = float(machine["motor_current"])
    features["rpm"] = float(machine["rpm"])
    features["operating_hours"] = float(machine["operating_hours"])
    features["workload"] = float(machine["workload"])

    # Rolling features
    for sensor in [
        "temperature",
        "vibration",
        "motor_current",
        "rpm"
    ]:

        values = window[sensor].values

        features[f"{sensor}_avg"] = float(np.mean(values))

        features[f"{sensor}_std"] = float(
            np.std(values)
        )

        features[f"{sensor}_slope"] = calculate_slope(values)

    features["type_code"] = TYPE_CODE[
        machine["machine_type"]
    ]

    return pd.DataFrame(
        [[features[f] for f in FEATURES]],
        columns=FEATURES
    )


# ============================================================
# RUN ML MODELS
# ============================================================

def run_prediction(machine_id):

    machine = machines[machine_id]

    X = create_features(machine_id)

    # ------------------------------------
    # ANOMALY
    # ------------------------------------

    anomaly_prediction = anomaly_model.predict(X)[0]

    anomaly_decision = anomaly_model.decision_function(X)[0]

    anomaly = anomaly_prediction == -1

    # Convert Isolation Forest score into
    # an easier 0-100 anomaly risk
    anomaly_risk = float(
        np.clip(
            (0.5 - anomaly_decision) * 100,
            0,
            100
        )
    )

    # ------------------------------------
    # CONDITION
    # ------------------------------------

    condition_prediction = condition_model.predict(X)[0]

    condition = str(condition_prediction)

    # ------------------------------------
    # RUL
    # ------------------------------------

    rul_prediction = rul_model.predict(X)[0]

    rul = max(float(rul_prediction), 0)

    # ------------------------------------
    # RISK
    # ------------------------------------

    if condition.lower() == "critical":
        condition_risk = 90

    elif condition.lower() == "warning":
        condition_risk = 60

    else:
        condition_risk = 20

    if rul <= 24:
        rul_risk = 90

    elif rul <= 72:
        rul_risk = 65

    elif rul <= 168:
        rul_risk = 40

    else:
        rul_risk = 15

    risk = (
        condition_risk * 0.45
        + anomaly_risk * 0.30
        + rul_risk * 0.25
    )

    risk = float(np.clip(risk, 0, 100))

    return {
        "machine_id": machine_id,
        "machine_type": machine["machine_type"],

        "temperature": round(machine["temperature"], 2),
        "vibration": round(machine["vibration"], 3),
        "motor_current": round(machine["motor_current"], 2),
        "rpm": round(machine["rpm"], 0),
        "workload": round(machine["workload"], 1),
        "operating_hours": round(
            machine["operating_hours"], 1
        ),

        "condition": condition,

        "anomaly": bool(anomaly),

        "anomaly_risk": round(
            anomaly_risk,
            2
        ),

        "rul_hours": round(
            rul,
            2
        ),

        "risk_score": round(
            risk,
            2
        ),

        "timestamp": time.strftime(
            "%H:%M:%S"
        ),

        "scenario": machine["scenario"],
    }


# ============================================================
# SIMULATOR
# ============================================================

def simulate_machine(machine_id):

    machine = machines[machine_id]

    scenario = machine["scenario"]

    # ------------------------------------
    # NORMAL
    # ------------------------------------

    if scenario == "normal":

        machine["temperature"] += np.random.normal(0, 0.15)

        machine["vibration"] += np.random.normal(0, 0.025)

        machine["motor_current"] += np.random.normal(0, 0.08)

        machine["rpm"] += np.random.normal(0, 30)

        machine["workload"] += np.random.normal(0, 1)

    # ------------------------------------
    # DEGRADING
    # ------------------------------------

    elif scenario == "degrading":

        machine["temperature"] += (
            0.06 + np.random.normal(0, 0.15)
        )

        machine["vibration"] += (
            0.025 + np.random.normal(0, 0.025)
        )

        machine["motor_current"] += (
            0.018 + np.random.normal(0, 0.06)
        )

        machine["rpm"] += (
            -12 + np.random.normal(0, 25)
        )

        machine["workload"] += (
            np.random.normal(0, 0.8)
        )

    # ------------------------------------
    # CRITICAL
    # ------------------------------------

    elif scenario == "critical":

        machine["temperature"] += (
            0.15 + np.random.normal(0, 0.2)
        )

        machine["vibration"] += (
            0.07 + np.random.normal(0, 0.04)
        )

        machine["motor_current"] += (
            0.04 + np.random.normal(0, 0.08)
        )

        machine["rpm"] += (
            -25 + np.random.normal(0, 30)
        )

        machine["workload"] += (
            np.random.normal(0, 1)
        )

    # ------------------------------------
    # LIMITS
    # ------------------------------------

    machine["temperature"] = np.clip(
        machine["temperature"],
        35,
        100
    )

    machine["vibration"] = np.clip(
        machine["vibration"],
        0.3,
        8
    )

    machine["motor_current"] = np.clip(
        machine["motor_current"],
        3,
        20
    )

    machine["rpm"] = np.clip(
        machine["rpm"],
        500,
        8000
    )

    machine["workload"] = np.clip(
        machine["workload"],
        10,
        100
    )

    machine["operating_hours"] += (
        1 / 60
    )

    # Save history

    history[machine_id].append({
        "temperature": machine["temperature"],
        "vibration": machine["vibration"],
        "motor_current": machine["motor_current"],
        "rpm": machine["rpm"],
    })

    # Keep last 50 readings

    history[machine_id] = history[machine_id][-50:]


def simulation_loop():

    while True:

        for machine_id in machines:

            simulate_machine(machine_id)

        time.sleep(2)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="UPTime Predictive Maintenance"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/live")
def get_live_data():

    results = []

    for machine_id in machines:

        results.append(
            run_prediction(machine_id)
        )

    return {
        "machines": results,
        "timestamp": time.strftime(
            "%H:%M:%S"
        )
    }


@app.get("/api/machine/{machine_id}")
def get_machine(machine_id: str):

    if machine_id not in machines:

        return {
            "error": "Machine not found"
        }

    return run_prediction(machine_id)


@app.get("/api/history/{machine_id}")
def get_history(machine_id: str):

    if machine_id not in machines:

        return []

    output = []

    for item in history[machine_id]:

        output.append({
            "temperature": round(
                item["temperature"],
                2
            ),

            "vibration": round(
                item["vibration"],
                3
            ),

            "motor_current": round(
                item["motor_current"],
                2
            ),

            "rpm": round(
                item["rpm"],
                0
            ),
        })

    return output


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def frontend():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "index.html"
        )
    )


# Start simulation
threading.Thread(
    target=simulation_loop,
    daemon=True
).start()


print("")
print("====================================")
print(" UPTime Predictive Maintenance")
print("====================================")
print("")
print("Dashboard:")
print("http://localhost:8000")
print("")
print("API:")
print("http://localhost:8000/api/live")
print("")