import { useEffect, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";

import {
    getDashboard,
    getMachines,
    getMachine,
    getMachineHistory,
    getAlerts,
} from "./api";

import "./App.css";

function App() {
    const [dashboard, setDashboard] = useState(null);
    const [machines, setMachines] = useState([]);
    const [selectedMachine, setSelectedMachine] = useState(null);
    const [history, setHistory] = useState([]);
    const [alerts, setAlerts] = useState([]);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        loadDashboard();
    }, []);

    async function loadDashboard() {
        try {
            setLoading(true);

            const [dashboardData, machinesData, alertsData] =
                await Promise.all([
                    getDashboard(),
                    getMachines(),
                    getAlerts(),
                ]);

            setDashboard(dashboardData);
            setMachines(machinesData);
            setAlerts(alertsData);

            if (machinesData.length > 0) {
                await selectMachine(machinesData[0].machine_id);
            }
        } catch (err) {
            console.error(err);
            setError(
                "Unable to connect to backend. Make sure FastAPI is running on port 8000."
            );
        } finally {
            setLoading(false);
        }
    }

    async function selectMachine(machineId) {
        try {
            const [machineData, historyData] = await Promise.all([
                getMachine(machineId),
                getMachineHistory(machineId),
            ]);

            setSelectedMachine(machineData);
            setHistory(historyData);
        } catch (err) {
            console.error(err);
        }
    }

    if (loading) {
        return (
            <div className="loading-screen">
                <div className="loader"></div>
                <p>Loading predictive maintenance system...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="error-screen">
                <h2>Backend Connection Error</h2>
                <p>{error}</p>

                <button onClick={loadDashboard}>Retry</button>
            </div>
        );
    }

    const totalMachines = dashboard?.total_machines || machines.length;

    const normal =
        dashboard?.normal ??
        machines.filter((m) => m.condition === "Normal").length;

    const warning =
        dashboard?.warning ??
        machines.filter((m) => m.condition === "Warning").length;

    const critical =
        dashboard?.critical ??
        machines.filter((m) => m.condition === "Critical").length;

    return (
        <div className="app">

            {/* SIDEBAR */}

            <aside className="sidebar">

                <div className="logo">
                    <div className="logo-icon">U</div>

                    <div>
                        <h2>UPTIME</h2>
                        <span>Predictive AI</span>
                    </div>
                </div>

                <div className="nav-section">

                    <p className="nav-label">MONITORING</p>

                    <button className="nav-item active">
                        <span>▦</span>
                        Dashboard
                    </button>

                    <button className="nav-item">
                        <span>⚙</span>
                        Machines
                    </button>

                    <button className="nav-item">
                        <span>⚠</span>
                        Alerts
                        {warning + critical > 0 && (
                            <span className="nav-badge">
                                {warning + critical}
                            </span>
                        )}
                    </button>

                    <button className="nav-item">
                        <span>◷</span>
                        Maintenance
                    </button>

                </div>

                <div className="sidebar-bottom">

                    <div className="system-status">
                        <span className="status-dot"></span>

                        <div>
                            <strong>System Online</strong>
                            <small>AI models active</small>
                        </div>
                    </div>

                    <div className="user-card">
                        <div className="avatar">A</div>

                        <div>
                            <strong>Admin</strong>
                            <small>Operator</small>
                        </div>
                    </div>

                </div>

            </aside>


            {/* MAIN */}

            <main className="main">

                {/* HEADER */}

                <header className="topbar">

                    <div>
                        <h1>Predictive Maintenance</h1>

                        <p>
                            Real-time machine health monitoring & failure prediction
                        </p>
                    </div>

                    <div className="header-right">

                        <div className="live-status">
                            <span></span>
                            LIVE
                        </div>

                        <button className="refresh-btn" onClick={loadDashboard}>
                            ↻
                        </button>

                    </div>

                </header>


                {/* OVERVIEW */}

                <section className="stats-grid">

                    <StatCard
                        title="Total Machines"
                        value={totalMachines}
                        subtitle="Machines monitored"
                        icon="▦"
                        type="blue"
                    />

                    <StatCard
                        title="Normal"
                        value={normal}
                        subtitle="Operating normally"
                        icon="✓"
                        type="green"
                    />

                    <StatCard
                        title="Warning"
                        value={warning}
                        subtitle="Needs attention"
                        icon="!"
                        type="yellow"
                    />

                    <StatCard
                        title="Critical"
                        value={critical}
                        subtitle="Immediate action"
                        icon="⚠"
                        type="red"
                    />

                </section>


                {/* CONTENT */}

                <section className="content-grid">

                    {/* MACHINE LIST */}

                    <div className="panel machines-panel">

                        <div className="panel-header">

                            <div>
                                <h2>Machine Status</h2>
                                <p>Current health overview</p>
                            </div>

                            <span className="count">
                                {machines.length} Machines
                            </span>

                        </div>


                        <div className="machine-list">

                            {machines.map((machine) => (

                                <div
                                    key={machine.machine_id}
                                    className={`machine-item ${selectedMachine?.machine_id === machine.machine_id
                                            ? "selected"
                                            : ""
                                        }`}
                                    onClick={() =>
                                        selectMachine(machine.machine_id)
                                    }
                                >

                                    <div className="machine-icon">
                                        ⚙
                                    </div>

                                    <div className="machine-info">

                                        <strong>{machine.machine_id}</strong>

                                        <span>{machine.machine_type}</span>

                                    </div>

                                    <div className="machine-right">

                                        <ConditionBadge
                                            condition={machine.condition}
                                        />

                                        <span className="risk-small">
                                            {Number(machine.risk_score || 0).toFixed(0)}%
                                        </span>

                                    </div>

                                </div>

                            ))}

                        </div>

                    </div>


                    {/* MACHINE DETAILS */}

                    {selectedMachine && (

                        <div className="panel details-panel">

                            <div className="panel-header">

                                <div>

                                    <div className="machine-title">

                                        <h2>
                                            {selectedMachine.machine_id}
                                        </h2>

                                        <ConditionBadge
                                            condition={selectedMachine.condition}
                                        />

                                    </div>

                                    <p>
                                        {selectedMachine.machine_type}
                                    </p>

                                </div>

                                <div className="risk-display">

                                    <span>RISK SCORE</span>

                                    <strong>
                                        {Number(
                                            selectedMachine.risk_score || 0
                                        ).toFixed(0)}
                                        %
                                    </strong>

                                </div>

                            </div>


                            {/* KPI */}

                            <div className="kpi-grid">

                                <Kpi
                                    title="Remaining Useful Life"
                                    value={`${Number(
                                        selectedMachine.rul_hours || 0
                                    ).toFixed(1)} h`}
                                    icon="◷"
                                />

                                <Kpi
                                    title="Temperature"
                                    value={`${Number(
                                        selectedMachine.temperature || 0
                                    ).toFixed(1)} °C`}
                                    icon="♨"
                                />

                                <Kpi
                                    title="Vibration"
                                    value={`${Number(
                                        selectedMachine.vibration || 0
                                    ).toFixed(2)}`}
                                    icon="≋"
                                />

                                <Kpi
                                    title="Health"
                                    value={`${(
                                        Number(selectedMachine.health || 0) * 100
                                    ).toFixed(1)}%`}
                                    icon="♥"
                                />

                            </div>


                            {/* CHART */}

                            <div className="chart-container">

                                <div className="chart-header">

                                    <div>
                                        <h3>Sensor Trends</h3>
                                        <p>Temperature & vibration history</p>
                                    </div>

                                    <div className="chart-legend">

                                        <span>
                                            <i className="legend-temp"></i>
                                            Temperature
                                        </span>

                                        <span>
                                            <i className="legend-vibration"></i>
                                            Vibration
                                        </span>

                                    </div>

                                </div>


                                <ResponsiveContainer width="100%" height={280}>

                                    <LineChart data={history}>

                                        <CartesianGrid
                                            strokeDasharray="3 3"
                                            stroke="#e8edf3"
                                        />

                                        <XAxis
                                            dataKey="timestamp"
                                            tick={{ fontSize: 11 }}
                                        />

                                        <YAxis
                                            yAxisId="left"
                                            tick={{ fontSize: 11 }}
                                        />

                                        <YAxis
                                            yAxisId="right"
                                            orientation="right"
                                            tick={{ fontSize: 11 }}
                                        />

                                        <Tooltip />

                                        <Line
                                            yAxisId="left"
                                            type="monotone"
                                            dataKey="temperature"
                                            stroke="#2563eb"
                                            strokeWidth={2}
                                            dot={false}
                                        />

                                        <Line
                                            yAxisId="right"
                                            type="monotone"
                                            dataKey="vibration"
                                            stroke="#f59e0b"
                                            strokeWidth={2}
                                            dot={false}
                                        />

                                    </LineChart>

                                </ResponsiveContainer>

                            </div>


                            {/* SENSOR VALUES */}

                            <div className="sensor-grid">

                                <Sensor
                                    label="Motor Current"
                                    value={selectedMachine.motor_current}
                                    unit="A"
                                />

                                <Sensor
                                    label="RPM"
                                    value={selectedMachine.rpm}
                                    unit="RPM"
                                />

                                <Sensor
                                    label="Workload"
                                    value={selectedMachine.workload}
                                    unit="%"
                                />

                                <Sensor
                                    label="Operating Hours"
                                    value={selectedMachine.operating_hours}
                                    unit="h"
                                />

                            </div>


                            {/* AI PREDICTION */}

                            <div className="prediction-box">

                                <div className="prediction-icon">
                                    ✦
                                </div>

                                <div>

                                    <strong>
                                        AI Prediction
                                    </strong>

                                    <p>
                                        Current machine condition is{" "}
                                        <b>
                                            {selectedMachine.condition}
                                        </b>
                                        . Estimated remaining useful life is{" "}
                                        <b>
                                            {Number(
                                                selectedMachine.rul_hours || 0
                                            ).toFixed(1)}
                                            hours
                                        </b>
                                        .
                                    </p>

                                </div>

                            </div>

                        </div>

                    )}

                </section>


                {/* ALERTS */}

                <section className="panel alerts-panel">

                    <div className="panel-header">

                        <div>
                            <h2>Recent Alerts</h2>
                            <p>Machine conditions requiring attention</p>
                        </div>

                        <span className="count">
                            {alerts.length} Alerts
                        </span>

                    </div>


                    {alerts.length === 0 ? (

                        <div className="empty-alerts">
                            ✓ No active alerts
                        </div>

                    ) : (

                        <div className="alerts-list">

                            {alerts.slice(0, 5).map((alert, index) => (

                                <div
                                    className="alert-row"
                                    key={index}
                                >

                                    <div className="alert-symbol">
                                        ⚠
                                    </div>

                                    <div>

                                        <strong>
                                            {alert.machine_id}
                                        </strong>

                                        <p>
                                            {alert.message ||
                                                `Machine condition: ${alert.condition}`}
                                        </p>

                                    </div>

                                    <ConditionBadge
                                        condition={alert.condition}
                                    />

                                </div>

                            ))}

                        </div>

                    )}

                </section>

            </main>

        </div>
    );
}


/* COMPONENTS */


function StatCard({
    title,
    value,
    subtitle,
    icon,
    type,
}) {
    return (
        <div className="stat-card">

            <div className={`stat-icon ${type}`}>
                {icon}
            </div>

            <div>

                <span>{title}</span>

                <strong>{value}</strong>

                <small>{subtitle}</small>

            </div>

        </div>
    );
}


function Kpi({ title, value, icon }) {
    return (
        <div className="kpi">

            <div className="kpi-icon">
                {icon}
            </div>

            <div>

                <span>{title}</span>

                <strong>{value}</strong>

            </div>

        </div>
    );
}


function Sensor({ label, value, unit }) {
    return (
        <div className="sensor">

            <span>{label}</span>

            <strong>
                {Number(value || 0).toFixed(2)}
                <small>{unit}</small>
            </strong>

        </div>
    );
}


function ConditionBadge({ condition }) {

    const value = condition || "Normal";

    return (
        <span
            className={`condition ${value.toLowerCase()}`}
        >
            <i></i>
            {value}
        </span>
    );
}


export default App;