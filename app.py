# ╔══════════════════════════════════════════════════════════════╗
# ║   Transformer PM — Flask Backend                            ║
# ║   Render.com deployment                                     ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import json
import sqlite3
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DB_PATH     = "transformer.db"
MODEL_PATH  = "model/isolation_forest.pkl"
SCALER_PATH = "model/scaler.pkl"

ML_FEATURES = ["oil_temp", "winding_temp", "current", "vibration"]

OIL_NORMAL   = 60.0
OIL_WARNING  = 40.0
OIL_CRITICAL = 40.0

ANOMALY_THRESHOLD = 0.15

# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS readings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                oil_temp      REAL    NOT NULL,
                winding_temp  REAL    NOT NULL,
                current       REAL    NOT NULL,
                vibration     REAL    NOT NULL,
                oil_level     REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id     INTEGER NOT NULL,
                timestamp      TEXT    NOT NULL,
                anomaly_score  REAL    NOT NULL,
                is_anomaly     INTEGER NOT NULL,
                oil_severity   INTEGER NOT NULL,
                ml_severity    INTEGER NOT NULL,
                alert_severity INTEGER NOT NULL,
                fault_type     TEXT    NOT NULL,
                health_index   REAL    NOT NULL,
                model_version  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_meta (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at    TEXT    NOT NULL,
                n_samples     INTEGER NOT NULL,
                model_version TEXT    NOT NULL,
                contamination REAL    NOT NULL
            );
        """)
        db.commit()
    print("[DB] Initialised")

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model  = None
scaler = None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[MODEL] Loaded from disk")
    else:
        print("[MODEL] Not found — waiting for training")

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
def predict_anomaly(oil_temp, winding_temp, current, vibration):
    if model is None or scaler is None:
        return 0.0, False, 0, "NORMAL"

    X        = np.array([[oil_temp, winding_temp, current, vibration]])
    X_scaled = scaler.transform(X)

    raw_score     = model.score_samples(X_scaled)[0]
    anomaly_score = float(np.clip(-raw_score, 0, 1))
    is_anomaly    = anomaly_score > ANOMALY_THRESHOLD

    if anomaly_score > 0.60:
        ml_severity = 2
    elif anomaly_score > ANOMALY_THRESHOLD:
        ml_severity = 1
    else:
        ml_severity = 0

    fault_type = "NORMAL"
    if is_anomaly:
        X_mean    = scaler.mean_
        X_diff    = abs(X_scaled[0] - X_mean / (scaler.scale_ + 1e-9))
        worst_idx = int(np.argmax(X_diff))
        fault_map = {
            0: "OIL_OVERTEMP",
            1: "WINDING_OVERTEMP",
            2: "OVERCURRENT",
            3: "VIBRATION"
        }
        fault_type = fault_map.get(worst_idx, "UNKNOWN")

    return anomaly_score, is_anomaly, ml_severity, fault_type


def check_oil_level(oil_level):
    if oil_level < OIL_CRITICAL:
        return 2, "OIL_LOW_CRIT"
    elif oil_level < OIL_NORMAL:
        return 1, "OIL_LOW_WARN"
    return 0, "NORMAL"

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/log", methods=["POST"])
def log_reading():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    required = ["oil_temp", "winding_temp", "current", "vibration", "oil_level"]
    for field in required:
        if field not in data:
            return jsonify({"error": "Missing field: " + field}), 400

    oil_temp     = float(data["oil_temp"])
    winding_temp = float(data["winding_temp"])
    current      = float(data["current"])
    vibration    = float(data["vibration"])
    oil_level    = float(data["oil_level"])
    ts           = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    db  = get_db()
    cur = db.execute(
        "INSERT INTO readings (timestamp, oil_temp, winding_temp, current, vibration, oil_level) VALUES (?, ?, ?, ?, ?, ?)",
        (ts, oil_temp, winding_temp, current, vibration, oil_level)
    )
    db.commit()
    reading_id = cur.lastrowid

    anomaly_score, is_anomaly, ml_sev, fault_type = predict_anomaly(
        oil_temp, winding_temp, current, vibration)

    oil_sev, oil_fault = check_oil_level(oil_level)

    final_severity = max(ml_sev, oil_sev)
    if final_severity > 0 and oil_sev >= ml_sev:
        fault_type = oil_fault

    health_index = round((1.0 - anomaly_score) * 100.0, 2)

    meta = db.execute(
        "SELECT model_version FROM model_meta ORDER BY id DESC LIMIT 1"
    ).fetchone()
    model_version = meta["model_version"] if meta else "none"

    db.execute(
        """INSERT INTO predictions
           (reading_id, timestamp, anomaly_score, is_anomaly, oil_severity,
            ml_severity, alert_severity, fault_type, health_index, model_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (reading_id, ts, anomaly_score, int(is_anomaly), oil_sev,
         ml_sev, final_severity, fault_type, health_index, model_version)
    )
    db.commit()

    return jsonify({
        "reading_id":     reading_id,
        "anomaly_score":  anomaly_score,
        "is_anomaly":     is_anomaly,
        "alert_severity": final_severity,
        "fault_type":     fault_type,
        "health_index":   health_index
    }), 201


@app.route("/predictions", methods=["GET"])
def get_predictions():
    limit = int(request.args.get("limit", 10))
    db    = get_db()
    rows  = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return jsonify([dict(r) for r in rows]), 200


@app.route("/api/status", methods=["GET"])
def api_status():
    db   = get_db()
    n    = db.execute("SELECT COUNT(*) as c FROM readings").fetchone()["c"]
    meta = db.execute(
        "SELECT * FROM model_meta ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return jsonify({
        "total_readings": n,
        "model_ready":    model is not None,
        "model_meta":     dict(meta) if meta else None
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()}), 200


@app.route("/api/train", methods=["POST"])
def api_train():
    import base64
    global model, scaler

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    try:
        model_b64     = data["model_b64"]
        scaler_b64    = data["scaler_b64"]
        n_samples     = int(data["n_samples"])
        contamination = float(data["contamination"])
        model_version = data.get("model_version", "v1.0")

        os.makedirs("model", exist_ok=True)

        with open(MODEL_PATH,  "wb") as f:
            f.write(base64.b64decode(model_b64))
        with open(SCALER_PATH, "wb") as f:
            f.write(base64.b64decode(scaler_b64))

        load_model()

        db = get_db()
        db.execute(
            "INSERT INTO model_meta (trained_at, n_samples, model_version, contamination) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
             n_samples, model_version, contamination)
        )
        db.commit()

        return jsonify({
            "status":        "model loaded",
            "model_version": model_version,
            "n_samples":     n_samples
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard", methods=["GET"])
def dashboard():
    db = get_db()

    last = db.execute("""
        SELECT r.*, p.anomaly_score, p.alert_severity, p.fault_type, p.health_index
        FROM readings r
        LEFT JOIN predictions p ON p.reading_id = r.id
        ORDER BY r.id DESC LIMIT 1
    """).fetchone()

    recent = db.execute("""
        SELECT r.timestamp, r.oil_temp, r.winding_temp, r.current,
               r.vibration, r.oil_level, p.anomaly_score,
               p.alert_severity, p.fault_type, p.health_index
        FROM readings r
        LEFT JOIN predictions p ON p.reading_id = r.id
        ORDER BY r.id DESC LIMIT 20
    """).fetchall()

    n_total   = db.execute("SELECT COUNT(*) as c FROM readings").fetchone()["c"]
    n_anomaly = db.execute(
        "SELECT COUNT(*) as c FROM predictions WHERE is_anomaly=1"
    ).fetchone()["c"]

    last = dict(last) if last else {}
    rows = [dict(r) for r in recent]

    sev            = last.get("alert_severity", 0)
    severity_color = {0: "#00ff88", 1: "#ffaa00", 2: "#ff3333"}
    color          = severity_color.get(sev, "#00ff88")

    if sev == 2:
        status_label = "CRITICAL"
        status_icon  = "RED"
    elif sev == 1:
        status_label = "WARNING"
        status_icon  = "YELLOW"
    else:
        status_label = "NORMAL"
        status_icon  = "GREEN"

    fault       = last.get("fault_type", "—")
    health      = last.get("health_index", 100)
    ascore      = last.get("anomaly_score", 0) or 0
    model_status = "READY" if model else "WAITING FOR TRAINING"

    def row_html(r):
        sev_r  = r.get("alert_severity", 0) or 0
        sc     = round(r.get("anomaly_score") or 0, 4)
        ts     = str(r.get("timestamp", ""))[-8:]
        ot     = r.get("oil_temp", "")
        wt     = r.get("winding_temp", "")
        cu     = r.get("current", "")
        vb     = r.get("vibration", "")
        ol     = r.get("oil_level", "")
        ft     = r.get("fault_type", "—")
        hi     = round(r.get("health_index") or 100, 1)
        return (
            "<tr>"
            "<td>" + ts + "</td>"
            "<td>" + str(ot) + "</td>"
            "<td>" + str(wt) + "</td>"
            "<td>" + str(cu) + "</td>"
            "<td>" + str(vb) + "</td>"
            "<td>" + str(ol) + "</td>"
            "<td>" + str(sc) + "</td>"
            "<td>" + str(hi) + "%</td>"
            "<td>" + str(ft) + "</td>"
            "<td class=\"s" + str(sev_r) + "\">" + str(sev_r) + "</td>"
            "</tr>"
        )

    rows_html = "".join(row_html(r) for r in rows)

    html = (
        "<!DOCTYPE html>"
        "<html><head>"
        "<meta charset='UTF-8'>"
        "<meta http-equiv='refresh' content='5'>"
        "<title>Transformer PM</title>"
        "<style>"
        "* { box-sizing: border-box; margin: 0; padding: 0; }"
        "body { background: #0d1117; color: #e6edf3; font-family: monospace; padding: 20px; }"
        "h1 { color: #58a6ff; margin-bottom: 20px; font-size: 1.4em; }"
        ".grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }"
        ".card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }"
        ".card h3 { color: #8b949e; font-size: 0.8em; margin-bottom: 8px; }"
        ".card .val { font-size: 1.8em; font-weight: bold; }"
        ".status { background: #161b22; border: 2px solid " + color + "; border-radius: 8px; padding: 16px; margin-bottom: 20px; text-align: center; }"
        ".status h2 { color: " + color + "; font-size: 1.6em; }"
        "table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }"
        "th { background: #21262d; color: #8b949e; padding: 10px; font-size: 0.75em; text-align: left; }"
        "td { padding: 8px 10px; font-size: 0.78em; border-bottom: 1px solid #21262d; }"
        "tr:hover { background: #1c2128; }"
        ".s0 { color: #00ff88; } .s1 { color: #ffaa00; } .s2 { color: #ff3333; }"
        "</style></head><body>"
        "<h1>Transformer PM — Live Dashboard</h1>"
        "<div class='status'>"
        "<h2>[" + status_icon + "] " + status_label + " — " + fault + "</h2>"
        "<p>Health: " + str(round(health, 1)) + "% &nbsp;|&nbsp; "
        "Score: " + str(round(ascore, 4)) + " &nbsp;|&nbsp; "
        "Readings: " + str(n_total) + " &nbsp;|&nbsp; "
        "Anomalies: " + str(n_anomaly) + " &nbsp;|&nbsp; "
        "Model: " + model_status + "</p>"
        "</div>"
        "<div class='grid'>"
        "<div class='card'><h3>OIL TEMP</h3><div class='val'>" + str(last.get("oil_temp", "—")) + " C</div></div>"
        "<div class='card'><h3>WINDING TEMP</h3><div class='val'>" + str(last.get("winding_temp", "—")) + " C</div></div>"
        "<div class='card'><h3>CURRENT</h3><div class='val'>" + str(last.get("current", "—")) + " A</div></div>"
        "<div class='card'><h3>VIBRATION</h3><div class='val'>" + str(last.get("vibration", "—")) + " m/s2</div></div>"
        "<div class='card'><h3>OIL LEVEL</h3><div class='val'>" + str(last.get("oil_level", "—")) + " %</div></div>"
        "<div class='card'><h3>MODEL STATUS</h3><div class='val' style='font-size:1em'>" + model_status + "</div></div>"
        "</div>"
        "<table><tr>"
        "<th>TIME</th><th>OIL C</th><th>WIND C</th>"
        "<th>I(A)</th><th>VIB</th><th>OIL%</th>"
        "<th>SCORE</th><th>HEALTH</th><th>FAULT</th><th>SEV</th>"
        "</tr>"
        + rows_html +
        "</table>"
        "<p style='color:#8b949e; margin-top:10px; font-size:0.75em'>"
        "Auto-refresh 5s | UTC</p>"
        "</body></html>"
    )
    return html


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
init_db()
load_model()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
