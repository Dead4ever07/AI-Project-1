import io
import json
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request, Response, send_file

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _resolve_input_dir() -> Path:
    for candidate in (BASE_DIR / "input", Path.cwd() / "input"):
        if candidate.exists():
            return candidate
    return BASE_DIR / "input"

INPUT_DIR = _resolve_input_dir()

app = Flask(__name__)

_state_lock = threading.Lock()
_run_queue: queue.Queue = queue.Queue()
_run_active = False
_run_result: dict = {}
_active_module = None


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _emit(ev: dict):
    _run_queue.put(ev)


def _format_label(filename):
    name = filename.rsplit('.', 1)[0]
    return name.replace('_', ' ').title()

def _input_csvs():
    if not INPUT_DIR.exists():
        return []
    all_csv = sorted(p.name for p in INPUT_DIR.glob("*.csv"))
    data_files = [(n, _format_label(n)) for n in all_csv if "submission" not in n.lower() and "sample" not in n.lower()]
    if not data_files:
        data_files = [(n, _format_label(n)) for n in all_csv]
        
    return data_files


@app.route("/")
def index():
    data_files = _input_csvs()
    return render_template("index.html", data_files=data_files)


@app.route("/run", methods=["POST"])
def run_algorithm():
    global _run_active, _run_queue, _run_result

    with _state_lock:
        if _run_active:
            return jsonify({"error": "An algorithm is already running"}), 409

        body = request.get_json(force=True)
        data_name = body.get("data_file", "")
        algorithm = body.get("algorithm", "hill_climbing")

        data_path = INPUT_DIR / data_name

        try:
            with open(data_path, "r") as f:
                header = f.readline().strip()
                if "number_days" in header:
                    vals = f.readline().strip().split(",")
                    n_days  = int(vals[0])
                    max_occ = int(vals[1])
                    min_occ = int(vals[2])
                    data = pd.read_csv(data_path, skiprows=2, index_col="family_id")
                else:
                    n_days = 100
                    min_occ = 125
                    max_occ = 300
                    data = pd.read_csv(data_path, index_col="family_id")

            family_size = data["n_people"].values
            choices     = data[[f"choice_{i}" for i in range(10)]].values
            
            from algorithms.initial_solution import generate_initial_solution
            sub = generate_initial_solution(family_size, choices, n_days, min_occ, max_occ)
                
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        _run_queue = queue.Queue()
        _run_result.clear()
        _run_active = True

    def worker():
        global _run_active, _run_result, _active_module
        try:
            if algorithm == "hill_climbing":
                from algorithms import hill_climbing
                hill_climbing._callback = _emit
                hill_climbing._stop = False
                _active_module = hill_climbing
                result = hill_climbing.run(sub, family_size, choices, n_days, min_occ, max_occ)
            elif algorithm == "lp_solution":
                from algorithms import lp_solution
                lp_solution._callback = _emit
                lp_solution._stop = False
                _active_module = lp_solution
                result = lp_solution.run(sub, family_size, choices, n_days, min_occ, max_occ)
            elif algorithm == "simulated_annealing":
                from algorithms import simulated_annealing
                simulated_annealing._callback = _emit
                simulated_annealing._stop = False
                _active_module = simulated_annealing
                result = simulated_annealing.run(sub, family_size, choices, n_days, min_occ, max_occ)
            elif algorithm == "hybrid":
                from algorithms import hybrid_solution
                hybrid_solution._callback = _emit
                hybrid_solution._stop = False
                _active_module = hybrid_solution
                result = hybrid_solution.run(sub, family_size, choices, n_days, min_occ, max_occ)                
            elif algorithm == "genetic":
                from algorithms import genetic_algorithm
                genetic_algorithm._callback = _emit
                genetic_algorithm._stop = False
                _active_module = genetic_algorithm
                result = genetic_algorithm.run(sub, family_size, choices, n_days, min_occ, max_occ)          
            elif algorithm == "hill_climbing_random":
                from algorithms import random_start_hill_climbing
                random_start_hill_climbing._callback = _emit
                random_start_hill_climbing._stop = False
                _active_module = random_start_hill_climbing
                result = random_start_hill_climbing.run(sub, family_size, choices, n_days, min_occ, max_occ)


            assignment = result[0]
            with _state_lock:
                _run_result["assignment"]  = assignment.tolist()
                _run_result["choices"]     = choices.tolist()
                _run_result["family_size"] = family_size.tolist()
                _run_result["n_days"]      = n_days
                _run_result["min_occ"]     = min_occ
                _run_result["max_occ"]     = max_occ
        except Exception as exc:
            _run_queue.put({"type": "error", "message": str(exc)})
        finally:
            with _state_lock:
                _run_active = False
                _active_module = None

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                ev = _run_queue.get(timeout=30)
                yield f"data: {json.dumps(ev, cls=_NumpyEncoder)}\n\n"
                if ev.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/stop", methods=["POST"])
def stop_algorithm():
    with _state_lock:
        if not _run_active or _active_module is None:
            return jsonify({"status": "not running"}), 200
        _active_module._stop = True
    return jsonify({"status": "stop requested"})


@app.route("/data/occupancy")
def data_occupancy():
    with _state_lock:
        if "assignment" not in _run_result:
            return jsonify({"error": "No result available"}), 404
        assignment  = np.array(_run_result["assignment"])
        family_size = np.array(_run_result["family_size"])
        n_days  = _run_result["n_days"]
        min_occ = _run_result["min_occ"]
        max_occ = _run_result["max_occ"]

    occupancy = np.zeros(n_days + 1)
    for f, d in enumerate(assignment):
        occupancy[d] += family_size[f]
    occ = occupancy[1:]

    return jsonify({
        "days": list(range(1, n_days + 1)),
        "occupancy": occ.tolist(),
        "min_occ": min_occ,
        "max_occ": max_occ
    })


@app.route("/data/choices")
def data_choices():
    with _state_lock:
        if "assignment" not in _run_result:
            return jsonify({"error": "No result available"}), 404
        assignment = np.array(_run_result["assignment"])
        choices    = np.array(_run_result["choices"])

    rank_counts = np.zeros(11, dtype=int)
    for f, d in enumerate(assignment):
        rank = 10
        for r in range(10):
            if choices[f, r] == d:
                rank = r
                break
        rank_counts[rank] += 1

    top3  = int(rank_counts[:3].sum())
    top10 = int(rank_counts[3:10].sum())
    not10 = int(rank_counts[10])

    return jsonify({
        "rank_counts": rank_counts.tolist(),
        "pie_data": [top3, top10, not10]
    })


@app.route("/data/family_size")
def data_family_size():
    with _state_lock:
        if "assignment" not in _run_result:
            return jsonify({"error": "No result available"}), 404
        family_size = np.array(_run_result["family_size"])

    sizes, counts = np.unique(family_size, return_counts=True)

    return jsonify({
        "sizes": sizes.tolist(),
        "counts": counts.tolist()
    })


@app.route("/download")
def download():
    with _state_lock:
        if "assignment" not in _run_result:
            return jsonify({"error": "No result available"}), 404
        assignment = _run_result["assignment"]

    df = pd.DataFrame({"assigned_day": assignment})
    df.index.name = "family_id"
    buf = io.BytesIO()
    df.to_csv(buf)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name="result.csv")


if __name__ == "__main__":
    print("Starting Santa's Workshop Tour GUI...")
    print("Open http://127.0.0.1:8080 in your browser")
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
