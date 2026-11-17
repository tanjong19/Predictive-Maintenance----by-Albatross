const API_BASE = "http://127.0.0.1:8000/api";

export async function getDashboard() {
    const response = await fetch(`${API_BASE}/dashboard`);

    if (!response.ok) {
        throw new Error("Failed to fetch dashboard");
    }

    return response.json();
}

export async function getMachines() {
    const response = await fetch(`${API_BASE}/machines`);

    if (!response.ok) {
        throw new Error("Failed to fetch machines");
    }

    return response.json();
}

export async function getMachine(machineId) {
    const response = await fetch(
        `${API_BASE}/machines/${encodeURIComponent(machineId)}`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch machine");
    }

    return response.json();
}

export async function getMachineHistory(machineId) {
    const response = await fetch(
        `${API_BASE}/machines/${encodeURIComponent(machineId)}/history`
    );

    if (!response.ok) {
        throw new Error("Failed to fetch machine history");
    }

    return response.json();
}

export async function getAlerts() {
    const response = await fetch(`${API_BASE}/alerts`);

    if (!response.ok) {
        throw new Error("Failed to fetch alerts");
    }

    return response.json();
}