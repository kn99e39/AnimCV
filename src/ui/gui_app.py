"""Point-and-click GUI for the full pipeline (Architecture_v2.md section 6.2).

``ui/mapping_ui.py``'s docstring calls the side-by-side frame+bone-list
click workflow described in section 6.2 steps 3-5 "future work" and ships
a text-driven mapper instead. This module delivers that: a Tkinter app
(stdlib only, no new dependency) covering every CLI stage
(extract-frames..export-blender) as tabs, with the bone mapping tab
showing the reference frame and its detected landmarks as clickable dots
instead of typed ``landmark``/``direction`` commands.

``custom_point`` mapping has no click support here because it has none
downstream either: ``retarget/solver.py``'s ``solve_anchor_bone`` looks
up ``source_names[0]`` in the MotionGraph's semantic-landmark tracks,
and a custom point isn't one of those tracks (README.md's own
create-mapping section already documents this: "custom_point maps a
user-specified tracked point to this bone -- no actual tracking pipeline
for this exists yet"). The GUI's custom_point entry is a plain text
field for schema parity with the CLI grammar, not a working feature.

Every heavy/optional import (cv2, mmpose, pyassimp, bpy-via-subprocess)
stays inside the button-handler closures below, mirroring
``app/cli.py``'s per-command lazy imports, so opening the GUI itself
never requires any optional extra to be installed.
"""

from __future__ import annotations

import base64
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from pose.pose_types import PoseLandmark


def frame_index_from_path(frame_path: str) -> int:
    """Same convention as app/cli.py's _frame_index_from_path: frame
    files are named by zero-padded index, e.g. 00000.png."""
    stem = Path(frame_path).stem
    return int(stem) if stem.isdigit() else 0


def nearest_landmark(
    landmarks: dict[str, PoseLandmark], x: float, y: float, max_distance: float = 25.0
) -> str | None:
    """Nearest visible landmark to a canvas click, within max_distance
    pixels, or None if nothing visible is close enough."""
    best_name: str | None = None
    best_dist = max_distance
    for name, lm in landmarks.items():
        if not lm.visible:
            continue
        dist = ((lm.x - x) ** 2 + (lm.y - y) ** 2) ** 0.5
        if dist <= best_dist:
            best_name = name
            best_dist = dist
    return best_name


def describe_mapping_entry(entry) -> str:
    if entry.mapping_mode == "direction":
        return f"direction {entry.source_names[0]} -> {entry.source_names[1]}"
    if entry.mapping_mode == "landmark":
        return f"landmark {entry.source_names[0]}"
    return f"custom_point {entry.source_names[0]}"


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    """Largest scale factor <= 1.0 that fits (width, height) inside
    (max_width, max_height) preserving aspect ratio. Returns 1.0 for a
    degenerate (non-positive) source size -- nothing to scale."""
    if width <= 0 or height <= 0:
        return 1.0
    return min(1.0, max_width / width, max_height / height)


def clamp_frame_range(start: int, end: int, moved: str) -> tuple[int, int]:
    """Keep ``start <= end`` after one endpoint moves. ``moved`` names the
    one the user just dragged, and it wins: dragging start past end pushes
    end up to meet it; dragging end below start pulls start down to meet
    it -- so the handle under the cursor never jumps away from it."""
    if start <= end:
        return start, end
    if moved == "start":
        return start, start
    return end, end


class MotionToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("AnimCV Motion Tool")
        root.geometry("1150x780")

        # Cross-tab convenience state -- each tab's fields are still
        # independently editable, this just pre-fills the likely next
        # input from the previous stage's output.
        self.rig_profile = None
        self.pose_sequence = None
        self.mapping_entries: list = []
        self.ik_chains: list = []
        self.selected_bone: str | None = None
        self.mapping_mode = tk.StringVar(value="landmark")
        self.pending_direction_first: str | None = None
        self.current_landmarks: dict[str, PoseLandmark] = {}
        self.mapping_frame_index = 0
        self._frame_photo = None  # keep a PhotoImage reference alive

        self.status_var = tk.StringVar(value="Ready")

        # Background work is only ever handed results back to the main
        # thread through this queue -- macOS's Tk is not reliably safe to
        # touch (not even via root.after()) from a non-main thread, so the
        # worker thread below never calls anything Tk-related itself.
        self._async_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self.root.after(50, self._poll_async_queue)

        ttk.Style().configure("Success.TLabel", foreground="#1a7f37", font=("TkDefaultFont", 10, "bold"))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self._build_frames_tab()
        self._build_pose_tab()
        self._build_rig_tab()
        self._build_mapping_tab()
        self._build_motion_tab()
        self._build_retarget_tab()
        self._build_optimize_tab()
        self._build_export_tab()

        bottom = ttk.Frame(root)
        bottom.pack(fill="x", side="bottom")
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor="w", padx=6)
        self.log_text = tk.Text(bottom, height=6, state="disabled")
        self.log_text.pack(fill="x", padx=6, pady=(0, 6))

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        # Release the scrubber's held VideoCapture before tearing down.
        if getattr(self, "_frames_scrubber", None) is not None:
            self._frames_scrubber.close()
            self._frames_scrubber = None
        self.root.destroy()

    # ---- generic helpers -------------------------------------------------

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_async_queue(self) -> None:
        try:
            while True:
                callback = self._async_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        self.root.after(50, self._poll_async_queue)

    def _run_async(
        self,
        work: Callable[[], object],
        on_success: Callable[[object], None] | None = None,
        busy_message: str = "Working...",
    ) -> None:
        """Runs work() off the Tk main thread so long operations
        (pose estimation, blender export) don't freeze the UI. The worker
        thread only ever puts a plain callable on _async_queue; the main
        thread's _poll_async_queue (scheduled via root.after from __init__,
        i.e. always from the main thread) is what actually touches Tk."""
        self.status_var.set(busy_message)

        def worker():
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
                self._async_queue.put(lambda: self._on_async_error(exc))
            else:
                self._async_queue.put(lambda: self._on_async_success(result, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _on_async_error(self, exc: Exception) -> None:
        self.status_var.set("Ready")
        self._log(f"ERROR: {exc}")
        messagebox.showerror("Error", str(exc))

    def _on_async_success(self, result, on_success) -> None:
        self.status_var.set("Ready")
        if on_success is not None:
            on_success(result)

    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        browse_kind: str = "open_file",
        filetypes: list[tuple[str, str]] | None = None,
        default: str = "",
    ) -> tk.StringVar:
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=var, width=55).grid(
            row=row, column=1, sticky="ew", padx=4, pady=3
        )

        def browse():
            if browse_kind == "open_file":
                path = filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
            elif browse_kind == "save_file":
                path = filedialog.asksaveasfilename(filetypes=filetypes or [("All files", "*.*")])
            else:
                path = filedialog.askdirectory()
            if path:
                var.set(path)

        ttk.Button(parent, text="Browse...", command=browse).grid(row=row, column=2, padx=4, pady=3)
        return var

    # ---- 1. extract-frames ------------------------------------------------

    _PREVIEW_W = 480
    _PREVIEW_H = 300

    def _build_frames_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="1. Frames")
        tab.columnconfigure(1, weight=1)

        # Scrubber state. The numeric start/end entries stay authoritative
        # for extraction (as before); the sliders + preview are an
        # optional visual way to set them, enabled only once a video is
        # loaded. _range_sync_guard stops the scale<->entry two-way sync
        # from recursing.
        self._frames_scrubber = None
        self._frames_preview_photo = None  # keep PhotoImage alive
        self._range_sync_guard = False
        self._preview_after_id = None
        self._pending_preview_index = 0

        self.video_path = self._path_row(
            tab, 0, "Video file:", "open_file", [("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")]
        )

        load_row = ttk.Frame(tab)
        load_row.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 0))
        ttk.Button(load_row, text="Load Preview", command=self._on_load_preview).pack(side="left")
        self.frames_preview_status = tk.StringVar(
            value="Load a video to scrub it and set the reference range visually (optional)."
        )
        ttk.Label(load_row, textvariable=self.frames_preview_status).pack(side="left", padx=(8, 0))

        self.frames_preview_canvas = tk.Canvas(
            tab,
            width=self._PREVIEW_W,
            height=self._PREVIEW_H,
            background="#222222",
            highlightthickness=1,
            highlightbackground="#444444",
        )
        self.frames_preview_canvas.grid(row=2, column=0, columnspan=3, padx=4, pady=4)
        self.frames_preview_info = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.frames_preview_info).grid(
            row=3, column=0, columnspan=3, sticky="w", padx=4
        )

        # Start: slider (0..last) <-> numeric entry, both drive the preview.
        ttk.Label(tab, text="Start frame:").grid(row=4, column=0, sticky="w", padx=4)
        self.start_scale = ttk.Scale(
            tab, from_=0, to=0, orient="horizontal", command=lambda v: self._on_range_scale("start", v)
        )
        self.start_scale.grid(row=4, column=1, sticky="ew", padx=4)
        self.start_scale.state(["disabled"])
        self.start_frame_var = tk.StringVar()
        start_entry = ttk.Entry(tab, textvariable=self.start_frame_var, width=8)
        start_entry.grid(row=4, column=2, padx=4)
        start_entry.bind("<Return>", lambda e: self._on_range_entry("start"))
        start_entry.bind("<FocusOut>", lambda e: self._on_range_entry("start"))

        ttk.Label(tab, text="End frame:").grid(row=5, column=0, sticky="w", padx=4)
        self.end_scale = ttk.Scale(
            tab, from_=0, to=0, orient="horizontal", command=lambda v: self._on_range_scale("end", v)
        )
        self.end_scale.grid(row=5, column=1, sticky="ew", padx=4)
        self.end_scale.state(["disabled"])
        self.end_frame_var = tk.StringVar()
        end_entry = ttk.Entry(tab, textvariable=self.end_frame_var, width=8)
        end_entry.grid(row=5, column=2, padx=4)
        end_entry.bind("<Return>", lambda e: self._on_range_entry("end"))
        end_entry.bind("<FocusOut>", lambda e: self._on_range_entry("end"))

        ttk.Label(
            tab, text="(blank = whole video; both are source-video frame indices, end inclusive)"
        ).grid(row=6, column=1, columnspan=2, sticky="w", padx=4)

        ttk.Label(tab, text="Target FPS (optional):").grid(row=7, column=0, sticky="w", padx=4)
        self.fps_var = tk.StringVar()
        ttk.Entry(tab, textvariable=self.fps_var, width=10).grid(row=7, column=1, sticky="w", padx=4)

        self.frames_out = self._path_row(tab, 8, "Output frames dir:", "dir")

        extract_row = ttk.Frame(tab)
        extract_row.grid(row=9, column=0, columnspan=3, sticky="w", padx=4, pady=8)
        ttk.Button(extract_row, text="Extract Frames", command=self._on_extract_frames).pack(
            side="left"
        )
        self.frames_progress_var = tk.StringVar(value="")
        ttk.Label(extract_row, textvariable=self.frames_progress_var).pack(side="left", padx=(10, 0))

        self.frames_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.frames_result_var, style="Success.TLabel").grid(
            row=10, column=0, columnspan=3, sticky="w", padx=4
        )

    @staticmethod
    def _safe_int(text: str, default: int, lo: int, hi: int) -> int:
        try:
            value = int(float(text))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, value))

    def _on_load_preview(self) -> None:
        video = self.video_path.get().strip()
        if not video:
            messagebox.showwarning("Missing input", "Choose a video file first.")
            return

        from mediaio.video_loader import VideoLoader

        if self._frames_scrubber is not None:
            self._frames_scrubber.close()
            self._frames_scrubber = None
        try:
            scrubber = VideoLoader().open_scrubber(video)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Could not open video", str(exc))
            return
        self._frames_scrubber = scrubber

        last = max(scrubber.frame_count - 1, 0)
        self._range_sync_guard = True
        for scale in (self.start_scale, self.end_scale):
            scale.configure(from_=0, to=last)
            scale.state(["!disabled"])
        start = self._safe_int(self.start_frame_var.get(), 0, 0, last)
        end = self._safe_int(self.end_frame_var.get(), last, 0, last)
        if not self.end_frame_var.get().strip():
            end = last
        self.start_scale.set(start)
        self.end_scale.set(end)
        self.start_frame_var.set(str(start))
        self.end_frame_var.set(str(end))
        self._range_sync_guard = False

        self.frames_preview_status.set(
            f"✓ {scrubber.frame_count} frames @ {scrubber.fps:.2f} fps, "
            f"{scrubber.width}x{scrubber.height}"
        )
        self._render_preview_frame(start)

    def _on_range_scale(self, which: str, value: str) -> None:
        if self._range_sync_guard or self._frames_scrubber is None:
            return
        # ttk.Scale passes the moved handle's new value to its command;
        # use that for the moved handle rather than reading it back, and
        # read the other handle from its own scale.
        moved = int(round(float(value)))
        if which == "start":
            start, end = moved, int(round(float(self.end_scale.get())))
        else:
            start, end = int(round(float(self.start_scale.get()))), moved
        start, end = clamp_frame_range(start, end, which)
        self._range_sync_guard = True
        self.start_scale.set(start)
        self.end_scale.set(end)
        self.start_frame_var.set(str(start))
        self.end_frame_var.set(str(end))
        self._range_sync_guard = False
        self._request_preview(start if which == "start" else end)

    def _on_range_entry(self, which: str) -> None:
        if self._range_sync_guard or self._frames_scrubber is None:
            return
        var = self.start_frame_var if which == "start" else self.end_frame_var
        if not var.get().strip():
            return
        last = max(self._frames_scrubber.frame_count - 1, 0)
        value = self._safe_int(var.get(), 0, 0, last)
        self._range_sync_guard = True
        (self.start_scale if which == "start" else self.end_scale).set(value)
        start = int(round(float(self.start_scale.get())))
        end = int(round(float(self.end_scale.get())))
        start, end = clamp_frame_range(start, end, which)
        self.start_scale.set(start)
        self.end_scale.set(end)
        self.start_frame_var.set(str(start))
        self.end_frame_var.set(str(end))
        self._range_sync_guard = False
        self._request_preview(start if which == "start" else end)

    def _request_preview(self, index: int) -> None:
        # Scale motion fires this rapidly during a drag; coalesce into one
        # render per ~30ms so seek+decode+encode doesn't lag the handle.
        self._pending_preview_index = index
        if self._preview_after_id is None:
            self._preview_after_id = self.root.after(30, self._flush_preview)

    def _flush_preview(self) -> None:
        self._preview_after_id = None
        self._render_preview_frame(self._pending_preview_index)

    def _render_preview_frame(self, index: int) -> None:
        scrubber = self._frames_scrubber
        if scrubber is None:
            return
        import cv2

        canvas = self.frames_preview_canvas
        frame = scrubber.read_frame(index)
        if frame is None:
            canvas.delete("all")
            canvas.create_text(
                self._PREVIEW_W // 2,
                self._PREVIEW_H // 2,
                text=f"(no frame at index {index})",
                fill="#999999",
            )
            self.frames_preview_info.set("")
            return
        scale = fit_scale(frame.shape[1], frame.shape[0], self._PREVIEW_W, self._PREVIEW_H)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            return
        # Tk 8.6 PhotoImage reads PNG from base64 `data`, so no Pillow is
        # needed to show an in-memory OpenCV (BGR) frame -- imencode writes
        # correct colors from BGR (see cv2.imwrite's own convention).
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"))
        self._frames_preview_photo = photo
        canvas.delete("all")
        canvas.create_image(self._PREVIEW_W // 2, self._PREVIEW_H // 2, image=photo)
        timestamp = index / scrubber.fps if scrubber.fps > 0 else 0.0
        self.frames_preview_info.set(
            f"Frame {index} / {max(scrubber.frame_count - 1, 0)}   (t = {timestamp:.2f}s)"
        )

    def _on_extract_frames(self) -> None:
        video = self.video_path.get().strip()
        out = self.frames_out.get().strip()
        if not video or not out:
            messagebox.showwarning("Missing input", "Video file and output directory are required.")
            return

        # Warn before writing into a directory that already has files --
        # extraction overwrites frames of the same index and leaves any
        # extras behind, so an accidental reuse of an old cache dir is
        # worth confirming.
        out_dir = Path(out)
        if out_dir.exists() and any(out_dir.iterdir()):
            if not messagebox.askyesno(
                "Output folder not empty",
                f"'{out}' already contains files. Extracting here may overwrite existing "
                "frames and mix old ones in. Continue?",
            ):
                return

        fps_text = self.fps_var.get().strip()
        fps = float(fps_text) if fps_text else None
        start_text = self.start_frame_var.get().strip()
        end_text = self.end_frame_var.get().strip()
        start_frame = int(start_text) if start_text else None
        end_frame = int(end_text) if end_text else None

        self.frames_progress_var.set("preparing…")

        def report_progress(done: int, total: int) -> None:
            # Worker thread -> main thread, same queue the async result
            # uses (drained by _poll_async_queue); never touch Tk here.
            self._async_queue.put(lambda: self.frames_progress_var.set(f"{done}/{total}"))

        def work():
            import cv2

            from common.serialization import write_json
            from mediaio.frame_sequence import FrameSequenceMetadata
            from mediaio.video_loader import VideoLoader

            sequence = VideoLoader().load_video(
                video, target_fps=fps, start_frame=start_frame, end_frame=end_frame
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            total = len(sequence.frames)
            # Cap UI updates to ~100 over the whole run so a long
            # extraction doesn't flood the queue with per-frame updates.
            step = max(1, total // 100)
            for done, frame in enumerate(sequence.frames, start=1):
                cv2.imwrite(str(out_dir / f"{frame.index:05d}.png"), frame.image)
                if done % step == 0 or done == total:
                    report_progress(done, total)
            write_json(
                out_dir / "metadata.json", FrameSequenceMetadata.from_sequence(sequence).to_dict()
            )
            return total

        def on_success(count: int):
            self.frames_result_var.set(f"✓ Extracted {count} frames -> {out}")
            self.frames_progress_var.set(f"{count}/{count}")
            self._log(f"[frames] extracted {count} frames to {out}")
            self.pose_frames_dir.set(out)
            pngs = sorted(Path(out).glob("*.png"))
            if pngs:
                self.mapping_frame_path.set(str(pngs[0]))

        self._run_async(work, on_success, "Extracting frames...")

    # ---- 2. estimate-pose ---------------------------------------------------

    def _build_pose_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="2. Pose")
        tab.columnconfigure(1, weight=1)

        self.pose_frames_dir = self._path_row(tab, 0, "Frames directory:", "dir")
        self.pose_config_path = self._path_row(
            tab, 1, "MMPose config:", "open_file", [("Python config", "*.py"), ("All", "*.*")]
        )
        self.pose_checkpoint_path = self._path_row(
            tab, 2, "MMPose checkpoint:", "open_file", [("Checkpoint", "*.pth"), ("All", "*.*")]
        )
        default_row = ttk.Frame(tab)
        default_row.grid(row=3, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))
        ttk.Button(
            default_row, text="Use Default Model (RTMPose-tiny)", command=self._on_use_default_pose_model
        ).pack(side="left")
        self.pose_default_model_status_var = tk.StringVar()
        ttk.Label(default_row, textvariable=self.pose_default_model_status_var).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(tab, text="Device:").grid(row=4, column=0, sticky="w", padx=4)
        self.pose_device = tk.StringVar(value="cpu")
        ttk.Combobox(
            tab, textvariable=self.pose_device, values=["cpu", "cuda", "mps"], width=10, state="readonly"
        ).grid(row=4, column=1, sticky="w", padx=4)

        ttk.Label(tab, text="Visibility threshold:").grid(row=5, column=0, sticky="w", padx=4)
        self.pose_visibility = tk.StringVar(value="0.3")
        ttk.Entry(tab, textvariable=self.pose_visibility, width=10).grid(
            row=5, column=1, sticky="w", padx=4
        )

        self.pose_depth_checkpoint = self._path_row(
            tab, 6, "Depth checkpoint (optional):", "open_file", [("Checkpoint", "*.pth"), ("All", "*.*")]
        )
        ttk.Label(tab, text="Depth encoder:").grid(row=7, column=0, sticky="w", padx=4)
        self.pose_depth_encoder = tk.StringVar(value="vits")
        ttk.Combobox(
            tab,
            textvariable=self.pose_depth_encoder,
            values=["vits", "vitb", "vitl", "vitg"],
            width=10,
            state="readonly",
        ).grid(row=7, column=1, sticky="w", padx=4)
        ttk.Label(tab, text="Depth device:").grid(row=8, column=0, sticky="w", padx=4)
        self.pose_depth_device = tk.StringVar(value="auto")
        ttk.Entry(tab, textvariable=self.pose_depth_device, width=10).grid(
            row=8, column=1, sticky="w", padx=4
        )

        self.pose_out = self._path_row(
            tab, 9, "Output pose.json:", "save_file", [("JSON", "*.json")]
        )

        ttk.Button(tab, text="Run Pose Estimation", command=self._on_estimate_pose).grid(
            row=10, column=1, sticky="w", pady=8
        )
        self.pose_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.pose_result_var, style="Success.TLabel").grid(
            row=11, column=0, columnspan=3, sticky="w", padx=4
        )

    def _on_use_default_pose_model(self) -> None:
        def work():
            from pose.default_model import get_default_pose_checkpoint_path, get_default_pose_config_path

            config_path = get_default_pose_config_path()
            checkpoint_path = get_default_pose_checkpoint_path()
            return config_path, checkpoint_path

        def on_success(result):
            config_path, checkpoint_path = result
            self.pose_config_path.set(config_path)
            self.pose_checkpoint_path.set(checkpoint_path)
            self.pose_default_model_status_var.set("✓ RTMPose-tiny ready")
            self._log(f"[pose] using default model: {config_path} / {checkpoint_path}")

        self._run_async(
            work, on_success, "Fetching default model (downloads ~13MB on first use)..."
        )

    def _on_estimate_pose(self) -> None:
        frames = self.pose_frames_dir.get().strip()
        config = self.pose_config_path.get().strip()
        checkpoint = self.pose_checkpoint_path.get().strip()
        out = self.pose_out.get().strip()
        if not all([frames, config, checkpoint, out]):
            messagebox.showwarning(
                "Missing input", "Frames dir, MMPose config, checkpoint, and output path are required."
            )
            return
        device = self.pose_device.get()
        visibility_threshold = float(self.pose_visibility.get() or 0.3)
        depth_checkpoint = self.pose_depth_checkpoint.get().strip() or None
        depth_encoder = self.pose_depth_encoder.get()
        depth_device = self.pose_depth_device.get() or "auto"

        def work():
            from common.serialization import write_json
            from mediaio.video_loader import VideoLoader
            from pose.mmpose_adapter import MMPoseConfig, PoseEstimator
            from pose.pose_types import PoseSequence

            sequence = VideoLoader().load_image_sequence(frames)
            pose_config = MMPoseConfig(
                config_path=config,
                checkpoint_path=checkpoint,
                device=device,
                visibility_threshold=visibility_threshold,
            )
            poses = PoseEstimator(pose_config).process_sequence(sequence)

            if depth_checkpoint:
                from pose.depth_estimator import DepthEstimator, DepthEstimatorConfig
                from pose.depth_sampling import sample_depth_at_landmarks

                depth_estimator = DepthEstimator(
                    DepthEstimatorConfig(
                        checkpoint_path=depth_checkpoint, encoder=depth_encoder, device=depth_device
                    )
                )
                depth_frames = []
                for frame, pose_frame in zip(sequence.frames, poses.frames):
                    depth_map = depth_estimator.infer_frame(frame.image)
                    depth_frames.append(sample_depth_at_landmarks(pose_frame, depth_map))
                poses = PoseSequence(
                    frames=depth_frames, source_fps=poses.source_fps, landmark_schema=poses.landmark_schema
                )

            write_json(out, poses.to_dict())
            return len(poses.frames)

        def on_success(count: int):
            self.pose_result_var.set(f"✓ Estimated pose for {count} frames -> {out}")
            self._log(f"[pose] estimated pose for {count} frames -> {out}")
            self.motion_pose_path.set(out)
            self.mapping_pose_path.set(out)

        self._run_async(work, on_success, "Running pose estimation (this can take a while)...")

    # ---- 3. parse-rig -----------------------------------------------------

    def _build_rig_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="3. Rig")
        tab.columnconfigure(1, weight=1)

        self.rig_path = self._path_row(
            tab, 0, "Rig file (.fbx etc.):", "open_file", [("FBX", "*.fbx"), ("All", "*.*")]
        )
        self.rig_out = self._path_row(
            tab, 1, "Output rig_profile.json:", "save_file", [("JSON", "*.json")]
        )

        ttk.Button(tab, text="Parse Rig", command=self._on_parse_rig).grid(
            row=2, column=1, sticky="w", pady=8
        )
        self.rig_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.rig_result_var, style="Success.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=4
        )

        ttk.Label(tab, text="Parsed bones:").grid(row=4, column=0, sticky="nw", padx=4, pady=(10, 0))
        self.rig_bone_list = tk.Listbox(tab, height=15)
        self.rig_bone_list.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        tab.rowconfigure(5, weight=1)

    def _on_parse_rig(self) -> None:
        rig = self.rig_path.get().strip()
        out = self.rig_out.get().strip()
        if not rig or not out:
            messagebox.showwarning("Missing input", "Rig file and output path are required.")
            return

        def work():
            from rig.rig_parser import RigParser
            from rig.rig_profile import save_rig_profile

            profile = RigParser().load(rig)
            save_rig_profile(profile, out)
            return profile

        def on_success(profile):
            self.rig_profile = profile
            self.rig_result_var.set(
                f"✓ Parsed {len(profile.bones)} bones (root={profile.root_bone!r}) -> {out}"
            )
            self._log(f"[rig] parsed {len(profile.bones)} bones from {rig} -> {out}")
            self.rig_bone_list.delete(0, "end")
            for name in sorted(profile.bones):
                self.rig_bone_list.insert("end", name)
            self.mapping_rig_path.set(rig)
            self.retarget_rig_path.set(rig)
            self.export_rig_path.set(rig)
            self._refresh_mapping_bone_list()
            self._refresh_ik_bone_options()

        self._run_async(work, on_success, "Parsing rig...")

    # ---- 4. create-mapping (click-based) -----------------------------------

    def _build_mapping_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="4. Mapping")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(3, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.columnconfigure(1, weight=1)
        self.mapping_rig_path = self._path_row(top, 0, "Rig file:", "open_file")
        self.mapping_frame_path = self._path_row(
            top, 1, "Reference frame image:", "open_file", [("PNG", "*.png")]
        )
        self.mapping_pose_path = self._path_row(
            top, 2, "Pose JSON (for landmark dots):", "open_file", [("JSON", "*.json")]
        )
        btn_row = ttk.Frame(top)
        btn_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Button(btn_row, text="Load Frame + Landmarks", command=self._on_load_mapping_frame).pack(
            side="left"
        )

        body = ttk.Frame(tab)
        body.grid(row=1, column=0, columnspan=3, rowspan=3, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="ns")
        ttk.Label(left, text="Rig bones (click to select):").pack(anchor="w")
        self.mapping_bone_tree = ttk.Treeview(
            left, columns=("mapping",), show="tree headings", height=22
        )
        self.mapping_bone_tree.heading("#0", text="Bone")
        self.mapping_bone_tree.heading("mapping", text="Assigned mapping")
        self.mapping_bone_tree.column("#0", width=140)
        self.mapping_bone_tree.column("mapping", width=220)
        self.mapping_bone_tree.pack(fill="y", expand=True)
        self.mapping_bone_tree.bind("<<TreeviewSelect>>", self._on_select_mapping_bone)

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)

        mode_row = ttk.Frame(right)
        mode_row.pack(anchor="w")
        for value, label in [
            ("landmark", "Landmark (click 1 point)"),
            ("direction", "Direction (click 2 points)"),
            ("custom_point", "Custom point (type id)"),
        ]:
            ttk.Radiobutton(
                mode_row, text=label, value=value, variable=self.mapping_mode, command=self._reset_pending_click
            ).pack(side="left", padx=4)

        self.mapping_status_var = tk.StringVar(value="Parse a rig and select a bone to begin.")
        ttk.Label(right, textvariable=self.mapping_status_var, foreground="blue").pack(anchor="w", pady=4)

        self.mapping_canvas = tk.Canvas(right, background="#222222", width=480, height=360)
        self.mapping_canvas.pack(fill="both", expand=True)
        self.mapping_canvas.bind("<Button-1>", self._on_mapping_canvas_click)

        custom_row = ttk.Frame(right)
        custom_row.pack(anchor="w", pady=4)
        ttk.Label(custom_row, text="Custom point id:").pack(side="left")
        self.custom_point_var = tk.StringVar()
        ttk.Entry(custom_row, textvariable=self.custom_point_var, width=20).pack(side="left", padx=4)
        ttk.Button(custom_row, text="Assign", command=self._on_assign_custom_point).pack(side="left")

        clear_row = ttk.Frame(right)
        clear_row.pack(anchor="w")
        ttk.Button(clear_row, text="Clear mapping for selected bone", command=self._on_clear_bone_mapping).pack(
            side="left"
        )

        ik_frame = ttk.LabelFrame(right, text="IK chains (optional, 2-bone: root -> mid -> end)")
        ik_frame.pack(fill="x", pady=8)
        self.ik_root_bone = tk.StringVar()
        self.ik_mid_bone = tk.StringVar()
        self.ik_end_bone = tk.StringVar()
        self.ik_root_source = tk.StringVar()
        self.ik_mid_source = tk.StringVar()
        self.ik_end_source = tk.StringVar()
        for i, (label, var) in enumerate(
            [
                ("root bone", self.ik_root_bone),
                ("mid bone", self.ik_mid_bone),
                ("end bone", self.ik_end_bone),
            ]
        ):
            ttk.Label(ik_frame, text=label).grid(row=0, column=i * 2, padx=2)
            cb = ttk.Combobox(ik_frame, textvariable=var, width=14, state="readonly")
            cb.grid(row=0, column=i * 2 + 1, padx=2)
            setattr(self, f"_ik_bone_combo_{i}", cb)
        for i, (label, var) in enumerate(
            [
                ("root src", self.ik_root_source),
                ("mid src", self.ik_mid_source),
                ("end src", self.ik_end_source),
            ]
        ):
            ttk.Label(ik_frame, text=label).grid(row=1, column=i * 2, padx=2)
            ttk.Entry(ik_frame, textvariable=var, width=14).grid(row=1, column=i * 2 + 1, padx=2)
        ttk.Button(ik_frame, text="Add IK Chain", command=self._on_add_ik_chain).grid(
            row=2, column=0, columnspan=6, pady=4
        )
        self.ik_chain_list = tk.Listbox(ik_frame, height=4)
        self.ik_chain_list.grid(row=3, column=0, columnspan=6, sticky="ew", padx=2)
        ttk.Button(ik_frame, text="Remove selected chain", command=self._on_remove_ik_chain).grid(
            row=4, column=0, columnspan=6, pady=2
        )

        save_row = ttk.Frame(tab)
        save_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        save_row.columnconfigure(0, weight=1)
        self.mapping_out = self._path_row(
            save_row, 0, "Save mapping to:", "save_file", [("JSON", "*.json")]
        )
        ttk.Button(save_row, text="Save Mapping", command=self._on_save_mapping).grid(
            row=1, column=1, sticky="w", pady=4
        )

    def _refresh_mapping_bone_list(self) -> None:
        # Deleting+reinserting items clears the Treeview's selection (and
        # queues a deferred <<TreeviewSelect>> that would reset
        # self.selected_bone to None the next time the event loop turns)
        # -- restore it afterwards so mapping a bone doesn't silently
        # deselect it.
        previously_selected = self.mapping_bone_tree.selection()
        self.mapping_bone_tree.delete(*self.mapping_bone_tree.get_children())
        if self.rig_profile is None:
            return
        for name in sorted(self.rig_profile.bones):
            entry = next((e for e in self.mapping_entries if e.target_bone == name), None)
            desc = describe_mapping_entry(entry) if entry else ""
            self.mapping_bone_tree.insert("", "end", iid=name, text=name, values=(desc,))
        if previously_selected and previously_selected[0] in self.rig_profile.bones:
            self.mapping_bone_tree.selection_set(previously_selected[0])

    def _refresh_ik_bone_options(self) -> None:
        if self.rig_profile is None:
            return
        names = sorted(self.rig_profile.bones)
        for i in range(3):
            getattr(self, f"_ik_bone_combo_{i}")["values"] = names

    def _on_select_mapping_bone(self, _event=None) -> None:
        selection = self.mapping_bone_tree.selection()
        self.selected_bone = selection[0] if selection else None
        self._reset_pending_click()

    def _reset_pending_click(self) -> None:
        self.pending_direction_first = None
        if self.selected_bone is None:
            self.mapping_status_var.set("Select a bone from the list first.")
        else:
            mode = self.mapping_mode.get()
            if mode == "direction":
                self.mapping_status_var.set(f"'{self.selected_bone}': click the FIRST landmark point.")
            elif mode == "landmark":
                self.mapping_status_var.set(f"'{self.selected_bone}': click a landmark point.")
            else:
                self.mapping_status_var.set(f"'{self.selected_bone}': type a point id and click Assign.")

    def _on_load_mapping_frame(self) -> None:
        frame_path = self.mapping_frame_path.get().strip()
        if not frame_path:
            messagebox.showwarning("Missing input", "Choose a reference frame image first.")
            return
        try:
            photo = tk.PhotoImage(file=frame_path)
        except tk.TclError as exc:
            messagebox.showerror("Could not load image", f"Only PNG frames are supported: {exc}")
            return

        self._frame_photo = photo  # keep alive -- PhotoImage is gc'd otherwise
        self.mapping_canvas.delete("all")
        self.mapping_canvas.config(width=photo.width(), height=photo.height())
        self.mapping_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.mapping_frame_index = frame_index_from_path(frame_path)

        self.current_landmarks = {}
        pose_path = self.mapping_pose_path.get().strip()
        if pose_path:
            from common.serialization import read_json
            from pose.pose_types import PoseSequence

            sequence = PoseSequence.from_dict(read_json(pose_path))
            matching = next(
                (f for f in sequence.frames if f.frame_index == self.mapping_frame_index), None
            )
            if matching is not None:
                self.current_landmarks = matching.landmarks
            else:
                self._log(
                    f"[mapping] no pose frame with index {self.mapping_frame_index} in {pose_path}"
                )

        for name, lm in self.current_landmarks.items():
            color = "#00ff66" if lm.visible else "#666666"
            self.mapping_canvas.create_oval(
                lm.x - 4, lm.y - 4, lm.x + 4, lm.y + 4, fill=color, outline="white", tags=("landmark",)
            )
            self.mapping_canvas.create_text(
                lm.x + 6, lm.y - 6, text=name, fill="white", anchor="w", font=("TkDefaultFont", 8)
            )
        self._log(f"[mapping] loaded frame {frame_path} with {len(self.current_landmarks)} landmarks")

    def _on_mapping_canvas_click(self, event: tk.Event) -> None:
        if self.selected_bone is None:
            messagebox.showinfo("No bone selected", "Select a bone from the list first.")
            return
        mode = self.mapping_mode.get()
        if mode == "custom_point":
            return  # handled by the text field + Assign button instead

        name = nearest_landmark(self.current_landmarks, event.x, event.y)
        if name is None:
            self.mapping_status_var.set("No landmark near that click -- try closer to a dot.")
            return

        if mode == "landmark":
            self._set_mapping_entry(self.selected_bone, "landmark", "landmark", [name])
            self.mapping_status_var.set(f"'{self.selected_bone}' -> landmark {name}")
            return

        # direction: needs two clicks
        if self.pending_direction_first is None:
            self.pending_direction_first = name
            self.mapping_status_var.set(f"First point: {name}. Now click the second point.")
        else:
            first = self.pending_direction_first
            self.pending_direction_first = None
            self._set_mapping_entry(self.selected_bone, "landmark", "direction", [first, name])
            self.mapping_status_var.set(f"'{self.selected_bone}' -> direction {first} -> {name}")

    def _on_assign_custom_point(self) -> None:
        if self.selected_bone is None:
            messagebox.showinfo("No bone selected", "Select a bone from the list first.")
            return
        point_id = self.custom_point_var.get().strip()
        if not point_id:
            messagebox.showwarning("Missing input", "Type a point id first.")
            return
        self._set_mapping_entry(self.selected_bone, "custom_point", "point", [point_id])
        self.mapping_status_var.set(f"'{self.selected_bone}' -> custom_point {point_id}")

    def _set_mapping_entry(
        self, bone: str, source_type: str, mapping_mode: str, source_names: list[str]
    ) -> None:
        from rig.bone_mapping import BoneMappingEntry

        self.mapping_entries = [e for e in self.mapping_entries if e.target_bone != bone]
        self.mapping_entries.append(
            BoneMappingEntry(
                target_bone=bone,
                source_type=source_type,
                source_names=source_names,
                mapping_mode=mapping_mode,
            )
        )
        self._refresh_mapping_bone_list()

    def _on_clear_bone_mapping(self) -> None:
        if self.selected_bone is None:
            return
        self.mapping_entries = [e for e in self.mapping_entries if e.target_bone != self.selected_bone]
        self._refresh_mapping_bone_list()
        self.mapping_status_var.set(f"Cleared mapping for '{self.selected_bone}'.")

    def _on_add_ik_chain(self) -> None:
        from rig.bone_mapping import IKChainEntry

        root_bone, mid_bone, end_bone = (
            self.ik_root_bone.get(),
            self.ik_mid_bone.get(),
            self.ik_end_bone.get(),
        )
        root_src, mid_src, end_src = (
            self.ik_root_source.get().strip(),
            self.ik_mid_source.get().strip(),
            self.ik_end_source.get().strip(),
        )
        if not all([root_bone, mid_bone, end_bone, root_src, mid_src, end_src]):
            messagebox.showwarning("Missing input", "All six IK chain fields are required.")
            return
        chain = IKChainEntry(
            name=f"ik_chain_{len(self.ik_chains) + 1}",
            root_bone=root_bone,
            mid_bone=mid_bone,
            end_bone=end_bone,
            root_source=root_src,
            mid_source=mid_src,
            end_source=end_src,
        )
        self.ik_chains.append(chain)
        self.ik_chain_list.insert(
            "end", f"{root_bone}->{mid_bone}->{end_bone}  ({root_src}/{mid_src}/{end_src})"
        )

    def _on_remove_ik_chain(self) -> None:
        selection = self.ik_chain_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.ik_chain_list.delete(index)
        del self.ik_chains[index]

    def _on_save_mapping(self) -> None:
        if self.rig_profile is None:
            messagebox.showwarning("No rig loaded", "Parse a rig first.")
            return
        out = self.mapping_out.get().strip()
        if not out:
            messagebox.showwarning("Missing input", "Choose where to save the mapping profile.")
            return

        from rig.bone_mapping import BoneMappingProfile, save_bone_mapping_profile

        profile = BoneMappingProfile(
            rig_id=self.rig_profile.rig_id,
            entries=list(self.mapping_entries),
            ik_chains=list(self.ik_chains),
            created_from_frame=self.mapping_frame_index,
        )
        save_bone_mapping_profile(profile, out)
        self._log(f"[mapping] saved {len(profile.entries)} entries -> {out}")
        self.retarget_mapping_path.set(out)

    # ---- 5. build-motion ----------------------------------------------------

    def _build_motion_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="5. Motion")
        tab.columnconfigure(1, weight=1)

        self.motion_pose_path = self._path_row(tab, 0, "Pose JSON:", "open_file", [("JSON", "*.json")])
        self.motion_out = self._path_row(
            tab, 1, "Output motion_graph.json:", "save_file", [("JSON", "*.json")]
        )
        ttk.Button(tab, text="Build Motion Graph", command=self._on_build_motion).grid(
            row=2, column=1, sticky="w", pady=8
        )
        self.motion_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.motion_result_var, style="Success.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=4
        )

    def _on_build_motion(self) -> None:
        pose_path = self.motion_pose_path.get().strip()
        out = self.motion_out.get().strip()
        if not pose_path or not out:
            messagebox.showwarning("Missing input", "Pose JSON and output path are required.")
            return

        def work():
            from common.serialization import read_json, write_json
            from motion.motion_builder import MotionGraphBuilder
            from pose.pose_types import PoseSequence

            poses = PoseSequence.from_dict(read_json(pose_path))
            graph = MotionGraphBuilder().build(poses, source_metadata={"pose_source": pose_path})
            write_json(out, graph.to_dict())
            return len(graph.frames)

        def on_success(count: int):
            self.motion_result_var.set(f"✓ Built motion graph with {count} frames -> {out}")
            self._log(f"[motion] built motion graph with {count} frames -> {out}")
            self.retarget_motion_path.set(out)

        self._run_async(work, on_success, "Building motion graph...")

    # ---- 6. retarget --------------------------------------------------------

    def _build_retarget_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="6. Retarget")
        tab.columnconfigure(1, weight=1)

        self.retarget_motion_path = self._path_row(
            tab, 0, "Motion graph JSON:", "open_file", [("JSON", "*.json")]
        )
        self.retarget_rig_path = self._path_row(tab, 1, "Rig file:", "open_file")
        self.retarget_mapping_path = self._path_row(
            tab, 2, "Mapping JSON:", "open_file", [("JSON", "*.json")]
        )
        self.retarget_out = self._path_row(
            tab, 3, "Output animation.json:", "save_file", [("JSON", "*.json")]
        )
        ttk.Button(tab, text="Retarget", command=self._on_retarget).grid(
            row=4, column=1, sticky="w", pady=8
        )
        self.retarget_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.retarget_result_var, style="Success.TLabel").grid(
            row=5, column=0, columnspan=3, sticky="w", padx=4
        )

    def _on_retarget(self) -> None:
        motion = self.retarget_motion_path.get().strip()
        rig = self.retarget_rig_path.get().strip()
        mapping = self.retarget_mapping_path.get().strip()
        out = self.retarget_out.get().strip()
        if not all([motion, rig, mapping, out]):
            messagebox.showwarning("Missing input", "All four fields are required.")
            return

        def work():
            from motion.motion_io import load_motion_graph
            from retarget.solver import RetargetSolver, save_animation_clip
            from rig.bone_mapping import load_bone_mapping_profile
            from rig.rig_parser import RigParser

            motion_graph = load_motion_graph(motion)
            rig_profile = RigParser().load(rig)
            mapping_profile = load_bone_mapping_profile(mapping)
            clip = RetargetSolver().solve(motion_graph, rig_profile, mapping_profile)
            save_animation_clip(clip, out)
            return len(clip.tracks)

        def on_success(count: int):
            self.retarget_result_var.set(f"✓ Retargeted {count} bone tracks -> {out}")
            self._log(f"[retarget] retargeted {count} bone tracks -> {out}")
            self.optimize_animation_path.set(out)

        self._run_async(work, on_success, "Retargeting...")

    # ---- 7. optimize --------------------------------------------------------

    def _build_optimize_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="7. Optimize")
        tab.columnconfigure(1, weight=1)

        self.optimize_animation_path = self._path_row(
            tab, 0, "Animation JSON:", "open_file", [("JSON", "*.json")]
        )
        ttk.Label(tab, text="Collapse preset:").grid(row=1, column=0, sticky="w", padx=4)
        self.optimize_collapse = tk.StringVar(value="medium")
        ttk.Combobox(
            tab,
            textvariable=self.optimize_collapse,
            values=["none", "light", "medium", "aggressive", "custom"],
            width=12,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(tab, text="Custom threshold:").grid(row=2, column=0, sticky="w", padx=4)
        self.optimize_threshold = tk.StringVar()
        ttk.Entry(tab, textvariable=self.optimize_threshold, width=10).grid(
            row=2, column=1, sticky="w", padx=4
        )
        self.optimize_out = self._path_row(
            tab, 3, "Output optimized animation.json:", "save_file", [("JSON", "*.json")]
        )
        ttk.Button(tab, text="Optimize", command=self._on_optimize).grid(
            row=4, column=1, sticky="w", pady=8
        )
        self.optimize_report = tk.Text(tab, height=10)
        self.optimize_report.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=4, pady=4)
        tab.rowconfigure(5, weight=1)

    def _on_optimize(self) -> None:
        animation = self.optimize_animation_path.get().strip()
        out = self.optimize_out.get().strip()
        if not animation or not out:
            messagebox.showwarning("Missing input", "Animation JSON and output path are required.")
            return
        preset = self.optimize_collapse.get()
        threshold_text = self.optimize_threshold.get().strip()
        threshold = float(threshold_text) if threshold_text else None
        if preset == "custom" and threshold is None:
            messagebox.showwarning("Missing input", "'custom' preset needs a threshold value.")
            return

        def work():
            from optimize.collapse import collapse_animation_clip
            from retarget.solver import load_animation_clip, save_animation_clip

            clip = load_animation_clip(animation)
            optimized_clip, reports = collapse_animation_clip(
                clip, preset=preset, custom_threshold=threshold
            )
            save_animation_clip(optimized_clip, out)
            return optimized_clip, reports

        def on_success(result):
            optimized_clip, reports = result
            self.optimize_report.delete("1.0", "end")
            for bone_name, report in reports.items():
                self.optimize_report.insert(
                    "end",
                    f"{bone_name}: {report.original_key_count} -> {report.optimized_key_count} keys "
                    f"(removed {report.removed_key_count}, max_error={report.max_error:.3f}, "
                    f"threshold={report.threshold:.3f})\n",
                )
            self._log(f"[optimize] optimized {len(optimized_clip.tracks)} bone tracks -> {out}")
            self.export_animation_path.set(out)

        self._run_async(work, on_success, "Optimizing...")

    # ---- 8. export-blender ---------------------------------------------------

    def _build_export_tab(self) -> None:
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="8. Export")
        tab.columnconfigure(1, weight=1)

        self.export_animation_path = self._path_row(
            tab, 0, "Optimized animation JSON:", "open_file", [("JSON", "*.json")]
        )
        self.export_rig_path = self._path_row(tab, 1, "Rig file:", "open_file")
        self.export_out = self._path_row(
            tab, 2, "Output .blend:", "save_file", [("Blender file", "*.blend")]
        )
        self.export_fbx_out = self._path_row(
            tab, 3, "Output .fbx (optional):", "save_file", [("FBX", "*.fbx")]
        )
        self.export_blender_exe = self._path_row(
            tab, 4, "Blender executable (optional override):", "open_file"
        )
        ttk.Button(tab, text="Export to Blender", command=self._on_export_blender).grid(
            row=5, column=1, sticky="w", pady=8
        )
        self.export_result_var = tk.StringVar()
        ttk.Label(tab, textvariable=self.export_result_var, style="Success.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w", padx=4
        )

    def _on_export_blender(self) -> None:
        animation = self.export_animation_path.get().strip()
        rig = self.export_rig_path.get().strip()
        out = self.export_out.get().strip()
        if not all([animation, rig, out]):
            messagebox.showwarning("Missing input", "Animation JSON, rig file, and output path are required.")
            return
        fbx_out = self.export_fbx_out.get().strip() or None
        blender_executable = self.export_blender_exe.get().strip() or None

        def work():
            from app.cli import run_export_blender

            run_export_blender(
                animation=animation, rig=rig, out=out, fbx_out=fbx_out, blender_executable=blender_executable
            )
            return None

        def on_success(_result):
            self.export_result_var.set(f"✓ Exported -> {out}")
            self._log(f"[export] exported Blender scene -> {out}")

        self._run_async(work, on_success, "Exporting to Blender (this can take a moment)...")


def main() -> None:
    root = tk.Tk()
    MotionToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
