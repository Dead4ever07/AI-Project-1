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


def _input_csvs():
    if not INPUT_DIR.exists():
        return [], []
    all_csv = sorted(p.name for p in INPUT_DIR.glob("*.csv"))
    data_files = [n for n in all_csv if "submission" not in n.lower() and "sample" not in n.lower()]
    sub_files  = [n for n in all_csv if "submission" in n.lower() or "sample" in n.lower()]
    if not data_files:
        data_files = all_csv
    if not sub_files:
        sub_files = all_csv
    return data_files, sub_files


@app.route("/")
def index():
    data_files, sub_files = _input_csvs()
    return render_template("index.html", data_files=data_files, sub_files=sub_files)


@app.route("/run", methods=["POST"])
def run_algorithm():
    global _run_active, _run_queue, _run_result

    with _state_lock:
        if _run_active:
            return jsonify({"error": "An algorithm is already running"}), 409

        body = request.get_json(force=True)
        data_name = body.get("data_file", "")
        sub_name  = body.get("sub_file", "")
        n_days    = int(body.get("n_days", 100))
        min_occ   = int(body.get("min_occ", 125))
        max_occ   = int(body.get("max_occ", 300))
        algorithm = body.get("algorithm", "hill_climbing")

        data_path = INPUT_DIR / data_name
        sub_path  = INPUT_DIR / sub_name

        try:
            data = pd.read_csv(data_path, index_col="family_id")
            sub  = pd.read_csv(sub_path,  index_col="family_id")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        family_size = data["n_people"].values
        choices     = data[[f"choice_{i}" for i in range(10)]].values

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
            else:
                from algorithms import simulated_annealing
                simulated_annealing._callback = _emit
                simulated_annealing._stop = False
                _active_module = simulated_annealing
                result = simulated_annealing.run(sub, family_size, choices, n_days, min_occ, max_occ)

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


@app.route("/graph/occupancy.png")
def graph_occupancy():
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

    fig, ax = plt.subplots(figsize=(10, 4))
    days = np.arange(1, n_days + 1)
    colors = ["#e05252" if (v < min_occ or v > max_occ) else "#4c9be8" for v in occ]
    ax.bar(days, occ, color=colors, alpha=0.85, width=0.85)
    ax.axhline(min_occ, color="#2ecc71", linestyle="--", linewidth=1.2, label=f"Min ({min_occ})")
    ax.axhline(max_occ, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Max ({max_occ})")
    ax.set_xlabel("Day", fontsize=11)
    ax.set_ylabel("People", fontsize=11)
    ax.set_title("Daily Workshop Occupancy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_days + 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/graph/choices.png")
def graph_choices():
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

    labels = [f"Choice {i}" for i in range(10)] + ["Not top-10"]
    palette = plt.cm.RdYlGn(np.linspace(0.85, 0.1, 11))

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 4.5))

    bars = ax_bar.bar(labels, rank_counts, color=palette, edgecolor="white", linewidth=0.6)
    for bar, cnt in zip(bars, rank_counts):
        if cnt:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                        str(cnt), ha="center", va="bottom", fontsize=8)
    ax_bar.set_xlabel("Preference Rank", fontsize=11)
    ax_bar.set_ylabel("Families", fontsize=11)
    ax_bar.set_title("Families per Choice Rank", fontsize=12, fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=45)

    top3  = rank_counts[:3].sum()
    top10 = rank_counts[3:10].sum()
    not10 = rank_counts[10]
    wedges, texts, autotexts = ax_pie.pie(
        [top3, top10, not10],
        labels=["Top-3 choices", "Choice 4-10", "Not in top-10"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax_pie.set_title("Preference Satisfaction", fontsize=12, fontweight="bold")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/graph/family_size.png")
def graph_family_size():
    with _state_lock:
        if "assignment" not in _run_result:
            return jsonify({"error": "No result available"}), 404
        family_size = np.array(_run_result["family_size"])

    sizes, counts = np.unique(family_size, return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes, counts, color="#9b59b6", alpha=0.85, edgecolor="white", width=0.6)
    ax.set_xlabel("Family Size (people)", fontsize=11)
    ax.set_ylabel("Number of Families", fontsize=11)
    ax.set_title("Family Size Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks(sizes)
    for s, c in zip(sizes, counts):
        ax.text(s, c + 0.3, str(c), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


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
