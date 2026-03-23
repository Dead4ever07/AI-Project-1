"""
Santa's Workshop Tour — Simple GUI
Run from project root: python gui.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import time

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


"""
The GUI should have multiple graphs that will be updated based on the data recived:

1º Graph:
    Total Cost / Iteration

2º Graph:
    Improvements / Iteration

3º Graph:
    Time / Iteration

Another graphs may be added in the future.

The algorithm should also provide a tab system or something similiar in order to
easly alternate between algorithms so that the comprarission is facilitated
"""


class SantaGUI:
    def __init__(self, root):
        self.root = root
        self.running = True
        self.root.title("Santa's Workshop Tour")
        self.root.geometry("800x600")

        self.q = queue.Queue()
        self.score_history = []
        self.last_assignment = None

        self._build()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._poll()

    def _build(self):
        # file picker
        file_frame = tk.Frame(self.root)
        file_frame.pack(fill="x", padx=16, pady=10)

        # File picker of the family information
        # Maybe in the future convert this to a selector with multiple data sizes
        tk.Label(file_frame, text="Family data:").grid(row=0, column=0, sticky="w")
        self.v_data = tk.StringVar(value="input/easy_input.csv")
        tk.Entry(file_frame, textvariable=self.v_data, width=45).grid(row=0, column=1, padx=6)
        tk.Button(file_frame, text="Browse",
                  command=lambda: self._browse(self.v_data)).grid(row=0, column=2)

        # Sample submission in the future will be related to the family data being possible to infer what values
        # would be used (perhapse all in the same file) 
        tk.Label(file_frame, text="Submission:").grid(row=1, column=0, sticky="w", pady=4)
        self.v_sub = tk.StringVar(value="input/easy_sample_sumbmission.csv")
        tk.Entry(file_frame, textvariable=self.v_sub, width=45).grid(row=1, column=1, padx=6)
        tk.Button(file_frame, text="Browse",
                  command=lambda: self._browse(self.v_sub)).grid(row=1, column=2)

        # 
        par_frame = tk.Frame(self.root)
        par_frame.pack(fill="x", padx=16)

        #Initial Values
        self.v_ndays = tk.IntVar(value=10)
        self.v_minoc = tk.IntVar(value=0)
        self.v_maxoc = tk.IntVar(value=20)

        for i, (label, var) in enumerate([
            ("N days", self.v_ndays),
            ("Min occupancy", self.v_minoc),
            ("Max occupancy", self.v_maxoc),
        ]):
            tk.Label(par_frame, text=label).grid(row=0, column=i*2, sticky="w", padx=(0, 4))
            tk.Entry(par_frame, textvariable=var, width=6).grid(row=0, column=i*2+1, padx=(0, 16))

        # buttons + status
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=16, pady=10)


        self.btn_run = tk.Button(btn_frame, text="Run", bg="#2ea043", fg="black",
                                 padx=12, command=self._start)

        self.btn_run.pack(side="left", padx=(0, 8))

        self.btn_save = tk.Button(btn_frame, text="Save result", state="disabled",
                                  padx=12, command=self._save)
        self.btn_save.pack(side="left")

        self.lbl_status = tk.Label(btn_frame, text="idle", fg="gray")
        self.lbl_status.pack(side="left", padx=16)

        self.lbl_score = tk.Label(btn_frame, text="", font=("TkDefaultFont", 12, "bold"))
        self.lbl_score.pack(side="right")

        # chart
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5))
        self.fig.tight_layout(pad=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=(0, 8))

        # log
        self.log = tk.Text(self.root, height=5, state="disabled", font=("Courier New", 9))
        self.log.pack(fill="x", padx=16, pady=(0, 12))

    def _browse(self, var):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p:
            var.set(p)

    # The log should not exist and the graph should dynamically show the progress via
    # a graph with total_cost/Iteration
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}]  {msg}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # The reading of the data should follow the previous comments
    def _start(self):
        try:
            data = pd.read_csv(self.v_data.get(), index_col="family_id")
            sub  = pd.read_csv(self.v_sub.get(),  index_col="family_id")
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        family_size = data["n_people"].values
        choices     = data[[f"choice_{i}" for i in range(10)]].values

        self.score_history.clear()
        self.last_assignment = None


        self.btn_run.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.lbl_status.configure(text="running…", fg="orange")
        self._log("Starting Hill Climbing…")

        n_days  = self.v_ndays.get()
        min_occ = self.v_minoc.get()
        max_occ = self.v_maxoc.get()

        def worker():
            from algorithms.hill_climbing import run
            run(sub, family_size, choices, n_days, min_occ, max_occ,
                callback=self.q.put)

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self):
        self.running = False
        plt.close(self.fig)
        self.root.destroy() 

    def _save(self):
        if self.last_assignment is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="result.csv",
        )
        if path:
            pd.DataFrame({"assigned_day": self.last_assignment}).to_csv(
                path, index_label="family_id")
            self._log(f"Saved → {path}")

    # The poll will be edited in the future to match other output formats
    def _poll(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass


        if(self.running):
            self.root.after(80, self._poll)

    def _handle(self, ev):
        t = ev["type"]

        if t == "start":
            self.score_history.append(ev["score"])
            self.lbl_score.configure(text=f"Cost: {ev['score']:,.2f}")
            self._log(f"Initial cost: {ev['score']:,.2f}  "
                      f"(pref {ev['preference']:,.2f} + acc {ev['accounting']:,.2f})")
            self._redraw()

        elif t == "iteration":
            self.score_history.append(ev["score"])
            self.lbl_score.configure(text=f"Cost: {ev['score']:,.2f}")
            self.lbl_status.configure(
                text=f"iter {ev['iteration']}  |  {ev['improvements']} moves")
            self._redraw()

        elif t == "done":
            self.last_assignment = ev["assignment"]
            self.score_history.append(ev["score"])
            self.btn_run.configure(state="normal")
            self.btn_save.configure(state="normal")
            self.lbl_status.configure(text="done ✓", fg="green")
            self.lbl_score.configure(text=f"Cost: {ev['score']:,.2f}")
            self._log(f"Done. Final cost: {ev['score']:,.2f}  "
                      f"(pref {ev['preference']:,.2f} + acc {ev['accounting']:,.2f})")
            self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.set_title("Cost over iterations")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Cost")

        if len(self.score_history) == 1:
            self.ax.plot([0], self.score_history, "o", color="#2ea043", markersize=8)
        elif len(self.score_history) > 1:
            xs = range(len(self.score_history))
            self.ax.plot(xs, self.score_history, color="#2ea043", linewidth=2)
            self.ax.fill_between(xs, self.score_history, alpha=0.15, color="#2ea043")

        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    SantaGUI(root)
    root.mainloop()