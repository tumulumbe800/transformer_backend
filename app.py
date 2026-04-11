import os
import sqlite3
import io
import base64
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
DB_PATH           = "/tmp/transformer.db"
ANOMALY_THRESHOLD = 0.62
ANOMALY_CRITICAL  = 0.72
OIL_NORMAL        = 60.0
OIL_CRITICAL      = 40.0

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
                timestamp     TEXT NOT NULL,
                oil_temp      REAL NOT NULL,
                winding_temp  REAL NOT NULL,
                current       REAL NOT NULL,
                vibration     REAL NOT NULL,
                oil_level     REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id     INTEGER NOT NULL,
                timestamp      TEXT NOT NULL,
                anomaly_score  REAL NOT NULL,
                is_anomaly     INTEGER NOT NULL,
                oil_severity   INTEGER NOT NULL,
                ml_severity    INTEGER NOT NULL,
                alert_severity INTEGER NOT NULL,
                fault_type     TEXT NOT NULL,
                health_index   REAL NOT NULL,
                model_version  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_store (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at    TEXT NOT NULL,
                n_samples     INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                contamination REAL NOT NULL,
                features      TEXT NOT NULL,
                model_blob    BLOB NOT NULL,
                scaler_blob   BLOB NOT NULL
            );
        """)
        db.commit()
    print("[DB] Initialised")

# ─────────────────────────────────────────────
#  MODEL — stored in DB, loaded into memory
# ─────────────────────────────────────────────
model    = None
scaler   = None
features = ["winding_temp", "current", "vibration"]

def load_model_from_db():
    global model, scaler, features
    try:
        with app.app_context():
            db  = get_db()
            row = db.execute(
                "SELECT * FROM model_store ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                model   = joblib.load(io.BytesIO(row["model_blob"]))
                scaler  = joblib.load(io.BytesIO(row["scaler_blob"]))
                features= row["features"].split(",")
                print(f"[MODEL] Loaded from DB — version={row['model_version']} features={features}")
            else:
                print("[MODEL] No model in DB yet")
    except Exception as e:
        print(f"[MODEL] Load error: {e}")

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
def predict_anomaly(row_data):
    if model is None or scaler is None:
        return 0.0, False, 0, "NORMAL"

    X        = np.array([[row_data[f] for f in features]])
    X_scaled = scaler.transform(X)

    raw_score     = model.score_samples(X_scaled)[0]
    anomaly_score = float(np.clip(-raw_score, 0, 1))
    is_anomaly    = anomaly_score > ANOMALY_THRESHOLD

    if anomaly_score > ANOMALY_CRITICAL:
        ml_severity = 2
    elif anomaly_score > ANOMALY_THRESHOLD:
        ml_severity = 1
    else:
        ml_severity = 0

    fault_type = "NORMAL"
    if is_anomaly:
        diffs     = np.abs(X_scaled[0])
        worst_idx = int(np.argmax(diffs))
        fault_map = {
            "oil_temp":     "OIL_OVERTEMP",
            "winding_temp": "WINDING_OVERTEMP",
            "current":      "OVERCURRENT",
            "vibration":    "VIBRATION"
        }
        fault_type = fault_map.get(features[worst_idx], "UNKNOWN")

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
    for f in required:
        if f not in data:
            return jsonify({"error": "Missing: " + f}), 400

    oil_temp     = float(data["oil_temp"])
    winding_temp = float(data["winding_temp"])
    current      = float(data["current"])
    vibration    = float(data["vibration"])
    oil_level    = float(data["oil_level"])
    ts           = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    db  = get_db()
    cur = db.execute(
        "INSERT INTO readings (timestamp,oil_temp,winding_temp,current,vibration,oil_level) VALUES (?,?,?,?,?,?)",
        (ts, oil_temp, winding_temp, current, vibration, oil_level)
    )
    db.commit()
    reading_id = cur.lastrowid

    row_data = {
        "oil_temp": oil_temp, "winding_temp": winding_temp,
        "current":  current,  "vibration":    vibration
    }
    anomaly_score, is_anomaly, ml_sev, fault_type = predict_anomaly(row_data)
    oil_sev, oil_fault = check_oil_level(oil_level)

    final_severity = max(ml_sev, oil_sev)
    if final_severity > 0 and oil_sev >= ml_sev:
        fault_type = oil_fault

    health_index  = round((1.0 - anomaly_score) * 100.0, 2)
    meta          = db.execute(
        "SELECT model_version FROM model_store ORDER BY id DESC LIMIT 1"
    ).fetchone()
    model_version = meta["model_version"] if meta else "none"

    db.execute(
        """INSERT INTO predictions
           (reading_id,timestamp,anomaly_score,is_anomaly,oil_severity,
            ml_severity,alert_severity,fault_type,health_index,model_version)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
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
        "SELECT id,trained_at,n_samples,model_version,contamination,features FROM model_store ORDER BY id DESC LIMIT 1"
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
    global model, scaler, features
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON"}), 400

    try:
        model_bytes  = base64.b64decode(data["model_b64"])
        scaler_bytes = base64.b64decode(data["scaler_b64"])
        n_samples    = int(data["n_samples"])
        contamination= float(data["contamination"])
        model_version= data.get("model_version", "v1.0")
        feat_list    = data.get("features", "winding_temp,current,vibration")

        # Load into memory
        model   = joblib.load(io.BytesIO(model_bytes))
        scaler  = joblib.load(io.BytesIO(scaler_bytes))
        features= feat_list.split(",") if isinstance(feat_list, str) else feat_list

        # Store in DB as blob — survives restarts
        db = get_db()
        db.execute(
            """INSERT INTO model_store
               (trained_at,n_samples,model_version,contamination,features,model_blob,scaler_blob)
               VALUES (?,?,?,?,?,?,?)""",
            (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
             n_samples, model_version, contamination,
             ",".join(features),
             sqlite3.Binary(model_bytes),
             sqlite3.Binary(scaler_bytes))
        )
        db.commit()
        print(f"[MODEL] Saved to DB — {model_version} features={features}")

        return jsonify({
            "status":        "model loaded",
            "model_version": model_version,
            "features":      features,
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

    sev   = last.get("alert_severity", 0) or 0
    color = {0: "#00ff88", 1: "#ffaa00", 2: "#ff3333"}.get(sev, "#00ff88")
    label = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}.get(sev, "NORMAL")
    fault = last.get("fault_type", "—") or "—"
    health= round(last.get("health_index") or 100, 1)
    ascore= round(last.get("anomaly_score") or 0, 4)
    mstat = "READY" if model else "WAITING FOR TRAINING"

    def make_row(r):
        s = r.get("alert_severity") or 0
        return (
            "<tr>"
            "<td>" + str(r.get("timestamp","")[-8:]) + "</td>"
            "<td>" + str(r.get("oil_temp","")) + "</td>"
            "<td>" + str(r.get("winding_temp","")) + "</td>"
            "<td>" + str(r.get("current","")) + "</td>"
            "<td>" + str(r.get("vibration","")) + "</td>"
            "<td>" + str(r.get("oil_level","")) + "</td>"
            "<td>" + str(round(r.get("anomaly_score") or 0, 4)) + "</td>"
            "<td>" + str(round(r.get("health_index") or 100, 1)) + "%</td>"
            "<td>" + str(r.get("fault_type","—")) + "</td>"
            "<td class='s" + str(s) + "'>" + str(s) + "</td>"
            "</tr>"
        )

    html = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='UTF-8'>"
        "<meta http-equiv='refresh' content='5'>"
        "<title>Transformer PM</title>"
        "<style>"
        "*{box-sizing:border-box;margin:0;padding:0}"
        "body{background:#0d1117;color:#e6edf3;font-family:monospace;padding:20px}"
        "h1{color:#58a6ff;margin-bottom:20px;font-size:1.4em}"
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:20px}"
        ".card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}"
        ".card h3{color:#8b949e;font-size:0.8em;margin-bottom:8px}"
        ".card .val{font-size:1.8em;font-weight:bold}"
        ".status{background:#161b22;border:2px solid " + color + ";border-radius:8px;padding:16px;margin-bottom:20px;text-align:center}"
        ".status h2{color:" + color + ";font-size:1.6em}"
        "table{width:100%;border-collapse:collapse;background:#161b22;border-radius:8px;overflow:hidden}"
        "th{background:#21262d;color:#8b949e;padding:10px;font-size:0.75em;text-align:left}"
        "td{padding:8px 10px;font-size:0.78em;border-bottom:1px solid #21262d}"
        "tr:hover{background:#1c2128}"
        ".s0{color:#00ff88}.s1{color:#ffaa00}.s2{color:#ff3333}"
        "</style></head><body>"
        "<h1>Transformer PM — Live Dashboard</h1>"
        "<div class='status'><h2>" + label + " — " + fault + "</h2>"
        "<p>Health: " + str(health) + "% | Score: " + str(ascore) +
        " | Readings: " + str(n_total) +
        " | Anomalies: " + str(n_anomaly) +
        " | Model: " + mstat + "</p></div>"
        "<div class='grid'>"
        "<div class='card'><h3>OIL TEMP</h3><div class='val'>" + str(last.get("oil_temp","—")) + " C</div></div>"
        "<div class='card'><h3>WINDING TEMP</h3><div class='val'>" + str(last.get("winding_temp","—")) + " C</div></div>"
        "<div class='card'><h3>CURRENT</h3><div class='val'>" + str(last.get("current","—")) + " A</div></div>"
        "<div class='card'><h3>VIBRATION</h3><div class='val'>" + str(last.get("vibration","—")) + " m/s2</div></div>"
        "<div class='card'><h3>OIL LEVEL</h3><div class='val'>" + str(last.get("oil_level","—")) + " %</div></div>"
        "<div class='card'><h3>MODEL</h3><div class='val' style='font-size:1em'>" + mstat + "</div></div>"
        "</div>"
        "<table><tr>"
        "<th>TIME</th><th>OIL C</th><th>WIND C</th>"
        "<th>I(A)</th><th>VIB</th><th>OIL%</th>"
        "<th>SCORE</th><th>HEALTH</th><th>FAULT</th><th>SEV</th>"
        "</tr>" + "".join(make_row(r) for r in rows) +
        "</table>"
        "<p style='color:#8b949e;margin-top:10px;font-size:0.75em'>Auto-refresh 5s | UTC</p>"
        "</body></html>"
    )
    return html
@app.route("/api/reload", methods=["POST"])
def api_reload():
    load_model_from_db()
    return jsonify({
        "model_ready": model is not None,
        "features":    features
    }), 200


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
init_db()
load_model_from_db()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
