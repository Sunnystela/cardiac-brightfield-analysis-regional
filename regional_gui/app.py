"""Regional rhythm viewer: inspect how patches were grouped and merge groups by hand.

    python launch.py regional-gui [--from-run <obs>/legacy_rhythm] [--input video --record records.json]

Flow: open a legacy_rhythm folder -> automatic groups (period + beat timing) are computed from the cache
(no optical flow) -> look at the map / traces / clustering tabs -> select 2+ groups -> Merge ->
everything is recomputed for the merged group -> Save writes regional_rhythm.json, regional_groups.csv,
the three figures and regional_merge.json (who/why note) into <observation>/regional_rhythm.

Merges are always stored in the AUTO label space (labels of groups_auto), so a saved merge file
reproduces the same result with `launch.py regional-rhythm --merge-file`.
"""
from __future__ import annotations

import queue
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

TABS = (("map", "Group map"), ("traces", "Traces / events"), ("clustering", "How grouped"))


class RegionalApp:
    def __init__(self, root: tk.Tk, from_run=None, video=None, record=None, mask_root=None):
        self.root = root
        root.title("Regional rhythm groups")
        root.geometry("1500x900")
        self.rr = None
        self.summary = None
        self.history: list[list[list[str]]] = []
        self.merges: list[list[str]] = []
        self.frame = self.mask = None
        self.preview_dir = Path(tempfile.mkdtemp(prefix="regional_preview_"))
        self.jobs: queue.Queue = queue.Queue()
        self.images = {}
        self.busy = False
        self.var_run = tk.StringVar(value=from_run or "")
        self.var_video = tk.StringVar(value=video or "")
        self.var_record = tk.StringVar(value=record or "")
        self.var_mask_root = tk.StringVar(value=mask_root or "")
        self.var_status = tk.StringVar(value="Open a legacy_rhythm folder.")
        self._build()
        root.after(150, self._poll)
        if from_run:
            self.load()

    # ------------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        for r, (label, var, cmd) in enumerate((("legacy_rhythm folder", self.var_run, self._browse_run),
                                               ("video (optional, map overlay)", self.var_video, self._browse_video),
                                               ("records JSON (optional)", self.var_record, self._browse_record))):
            ttk.Label(top, text=label, width=28).grid(row=r, column=0, sticky="w")
            ttk.Entry(top, textvariable=var, width=110).grid(row=r, column=1, sticky="we", padx=4)
            ttk.Button(top, text="...", width=3, command=cmd).grid(row=r, column=2)
        ttk.Button(top, text="Load / recompute", command=self.load).grid(row=0, column=3, rowspan=3, padx=8, sticky="ns")
        top.columnconfigure(1, weight=1)

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=6)
        body.add(left, weight=0)
        ttk.Label(left, text="Groups (Ctrl/Shift-click to select 2+)").pack(anchor="w")
        self.groups_box = tk.Listbox(left, selectmode="extended", width=58, height=16, font=("Consolas", 10),
                                     exportselection=False)
        self.groups_box.pack(fill="x")
        row = ttk.Frame(left)
        row.pack(fill="x", pady=4)
        for text, cmd in (("Merge selected", self.merge_selected), ("Undo", self.undo), ("Reset", self.reset),
                          ("Save", self.save)):
            ttk.Button(row, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(left, text="Group pairs (yellow = merge suggested: same rate, shape matches after lag)").pack(anchor="w", pady=(8, 0))
        self.pairs_box = tk.Listbox(left, width=58, height=14, font=("Consolas", 9), exportselection=False)
        self.pairs_box.pack(fill="both", expand=True)
        self.pairs_box.bind("<Double-Button-1>", self._select_pair)
        ttk.Label(left, text="Double-click a pair to select both groups.", foreground="gray").pack(anchor="w")
        self.merge_label = ttk.Label(left, text="manual merges: none", wraplength=420)
        self.merge_label.pack(anchor="w", pady=6)

        self.tabs = ttk.Notebook(body)
        body.add(self.tabs, weight=1)
        self.canvases = {}
        for key, title in TABS:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            canvas = tk.Canvas(frame, background="white", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda e, k=key: self._show(k))
            self.canvases[key] = canvas
        ttk.Label(self.root, textvariable=self.var_status, relief="sunken", anchor="w", padding=3).pack(fill="x")

    def _browse_run(self):
        d = filedialog.askdirectory(title="Select <observation>/legacy_rhythm")
        if d:
            self.var_run.set(d)

    def _browse_video(self):
        f = filedialog.askopenfilename(title="Video", filetypes=[("video", "*.avi *.mp4 *.mov"), ("all", "*.*")])
        if f:
            self.var_video.set(f)

    def _browse_record(self):
        f = filedialog.askopenfilename(title="segmentation_interface_records.json", filetypes=[("json", "*.json")])
        if f:
            self.var_record.set(f)

    # ------------------------------------------------------------------ background work
    def _run_bg(self, label, fn, done):
        if self.busy:  # engine modules are not thread-safe: one job at a time
            self.var_status.set("busy - wait for the current step to finish")
            return
        self.busy = True
        self.var_status.set(label + " ...")
        self.root.config(cursor="watch")

        def work():
            try:
                self.jobs.put((done, fn(), None))
            except Exception as exc:  # shown to the user, never swallowed
                import traceback
                self.jobs.put((done, None, f"{exc!r}\n\n{traceback.format_exc()}"))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, result, err = self.jobs.get_nowait()
                self.busy = False
                self.root.config(cursor="")
                if err:
                    self.var_status.set("error")
                    messagebox.showerror("Regional rhythm", err[:3000])
                else:
                    done(result)
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    # ------------------------------------------------------------------ actions
    def load(self):
        run = self.var_run.get().strip()
        if not run:
            messagebox.showinfo("Regional rhythm", "Choose a legacy_rhythm folder first.")
            return
        video, record = self.var_video.get().strip(), self.var_record.get().strip()

        def fn():
            from adapters.records import load_records
            from adapters.regional_rhythm import RegionalRhythm, overlay_inputs
            rr = RegionalRhythm(Path(run))
            frame = mask = None
            if video and record:
                recs = load_records(Path(record))
                rec = next((r for r in recs if _obs(r) == rr.cache["observation_id"]), recs[0] if len(recs) == 1 else None)
                if rec is None:
                    raise LookupError(f"no record for {rr.cache['observation_id']} in {record}")
                frame, mask = overlay_inputs(Path(video), rec, self.var_mask_root.get().strip() or Path(record).parent)
            return rr, frame, mask

        def done(res):
            self.rr, self.frame, self.mask = res
            self.merges, self.history = [], []
            self.refresh()
        self._run_bg("computing groups from the motion cache", fn, done)

    def refresh(self):
        merges = [list(m) for m in self.merges]

        def fn():
            from adapters.regional_rhythm_figures import render_all
            s = self.rr.summary(merges or None)
            figs = render_all(self.rr, s, self.preview_dir, self.frame, self.mask)
            return s, figs

        def done(res):
            self.summary, figs = res
            self.fig_paths = {"map": figs["map"]["path"], "traces": figs["traces"], "clustering": figs["clustering"]}
            self.images.clear()
            self._fill_lists()
            for k, _ in TABS:
                self._show(k)
            s = self.summary
            self.var_status.set(f"{s['observation_id']}: {s['status']} | groups: "
                                + ", ".join(f"{g['label']}={g['bpm']:.1f}" for g in s["groups_final"] if g["assigned"] and g["bpm"])
                                + f" | {s['n_reliable']}/{s['n_patches']} reliable patches | {figs['map']['note'] or 'map on frame'}")
        self._run_bg("recomputing groups and figures", fn, done)

    def _fill_lists(self):
        s = self.summary
        self.groups_box.delete(0, "end")
        self.final_groups = s["groups_final"]
        for g in self.final_groups:
            bpm = f"{g['bpm']:6.1f}" if g["bpm"] else "     -"
            frm = "+".join(g["merged_from"])
            self.groups_box.insert("end", f"{g['label']:<4}{bpm} BPM  n={g['n_patches']:<3} "
                                          f"IBI={_f(g['median_ibi_s'], 2):>5}s  from {frm}")
            if not g["assigned"]:
                self.groups_box.itemconfig("end", foreground="gray")
        self.pairs_box.delete(0, "end")
        self.pairs = sorted(s["pairs_final"], key=lambda p: (not p["merge_suggested"], -(p["lag_aligned_shape_correlation"] or -9)))
        for p in self.pairs:
            self.pairs_box.insert("end", f"{p['a']:>3}-{p['b']:<3} {_f(p['bpm_a'])}/{_f(p['bpm_b'])} BPM  "
                                         f"ratio x{p['harmonic_ratio'] or 0:g}  lag {_f(p['best_lag_s'], 2)}s  "
                                         f"shape {_f(p['lag_aligned_shape_correlation'], 2)}")
            if p["merge_suggested"]:
                self.pairs_box.itemconfig("end", background="#fff2b3")
        self.merge_label.config(text="manual merges (auto labels): " + (", ".join("+".join(m) for m in self.merges) or "none"))

    def _select_pair(self, _event):
        sel = self.pairs_box.curselection()
        if not sel:
            return
        p = self.pairs[sel[0]]
        self.groups_box.selection_clear(0, "end")
        for i, g in enumerate(self.final_groups):
            if g["label"] in (p["a"], p["b"]):
                self.groups_box.selection_set(i)

    def merge_selected(self):
        if not self.summary:
            return
        sel = [self.final_groups[i] for i in self.groups_box.curselection()]
        if len(sel) < 2:
            messagebox.showinfo("Merge", "Select at least two groups.")
            return
        new = sorted({a for g in sel for a in g["merged_from"]}, key=_label_key)
        self.history.append([list(m) for m in self.merges])
        self.merges = [m for m in self.merges if not set(m) & set(new)] + [new]
        self.refresh()

    def undo(self):
        if self.history:
            self.merges = self.history.pop()
            self.refresh()

    def reset(self):
        if self.merges:
            self.history.append([list(m) for m in self.merges])
            self.merges = []
            self.refresh()

    def save(self):
        if not self.summary:
            return
        default = self.rr.cache["legacy_dir"].parent / "regional_rhythm"
        out = filedialog.askdirectory(title="Save into (default: <observation>/regional_rhythm)", initialdir=str(default.parent))
        out = Path(out) if out else default
        note = simpledialog.askstring("Note", "Why were these groups merged? (saved in regional_merge.json)",
                                      parent=self.root) if self.merges else ""
        merges = [list(m) for m in self.merges]

        def fn():
            from adapters.regional_rhythm import write_merge_file, write_outputs
            from adapters.regional_rhythm_figures import render_all
            s = self.rr.summary(merges or None)
            paths = write_outputs(s, out)
            paths["figures"] = render_all(self.rr, s, out, self.frame, self.mask)
            paths["merge_file"] = str(write_merge_file(out / "regional_merge.json", merges, note or ""))
            return paths

        self._run_bg("saving", fn, lambda p: (self.var_status.set(f"saved to {out}"),
                                              messagebox.showinfo("Saved", "\n".join(str(v) for v in p.values()))))

    # ------------------------------------------------------------------ images
    def _show(self, key):
        path = getattr(self, "fig_paths", {}).get(key)
        canvas = self.canvases[key]
        if not path or not Path(path).is_file():
            return
        from PIL import Image, ImageTk
        w, h = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        img = Image.open(path)
        scale = min(w / img.width, h / img.height)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        self.images[key] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(w // 2, h // 2, image=self.images[key], anchor="center")


def _obs(rec) -> str:
    return f"{rec.batch}_{rec.condition}_{rec.well}_D{rec.day}_{rec.modality}"


def _label_key(label: str):
    return (label[0] != "G", int(label[1:]) if label[1:].isdigit() else 0)


def _f(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main(from_run=None, video=None, record=None, mask_root=None):
    root = tk.Tk()
    RegionalApp(root, from_run, video, record, mask_root)
    root.mainloop()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import adapters  # noqa: F401  (engine bootstrap before any cardio_organoid import)
    main(*(sys.argv[1:2] or [None]))
