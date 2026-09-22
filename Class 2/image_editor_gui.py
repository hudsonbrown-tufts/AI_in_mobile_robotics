"""Interactive GUI for binarization + binary morphology, with a live preview.

Usage:
    python image_editor_gui.py [path_or_url]

Pipeline (always applied in this fixed order, each step toggle-able):
    grayscale -> threshold (binarize) -> erosion -> dilation -> opening ->
    closing -> hit-or-miss -> boundary extraction -> skeletonize -> prune

Requires: numpy, opencv-python, scikit-image, pillow
"""
# run with: & "..\.venv\Scripts\python.exe" image_editor_gui.py WIN_20260908_11_39_47_Pro_grayscale.png


import sys
import tkinter as tk
from io import BytesIO
from tkinter import filedialog, messagebox, ttk
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk
from skimage.morphology import skeletonize, thin

MAX_PREVIEW_DIM = 900

SHAPES = {"Rectangle": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}

HITMISS_PRESETS = {
    "Isolated points": np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8),
    "Corner (top-left)": np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 1]], dtype=np.int8),
    "Corner (top-right)": np.array([[-1, -1, -1], [1, 1, -1], [1, 1, -1]], dtype=np.int8),
    "Corner (bottom-left)": np.array([[-1, 1, 1], [-1, 1, 1], [-1, -1, -1]], dtype=np.int8),
    "Corner (bottom-right)": np.array([[1, 1, -1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8),
}


def load_image_bgr(source: str) -> np.ndarray:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        pil_image = Image.open(source).convert("RGB")

    if max(pil_image.size) > MAX_PREVIEW_DIM:
        pil_image.thumbnail((MAX_PREVIEW_DIM, MAX_PREVIEW_DIM), Image.LANCZOS)

    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prune(binary01: np.ndarray, iterations: int) -> np.ndarray:
    """Iteratively remove endpoint pixels (1-neighbor spurs) from a skeleton."""
    result = binary01.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(iterations, 0)):
        neighbor_count = cv2.filter2D(result, ddepth=cv2.CV_8U, kernel=kernel) - result
        endpoints = (result == 1) & (neighbor_count == 1)
        if not np.any(endpoints):
            break
        result[endpoints] = 0
    return result


class MorphStep:
    def __init__(self, name, enabled_var, build_controls):
        self.name = name
        self.enabled_var = enabled_var
        self.build_controls = build_controls


class ImageEditorApp:
    def __init__(self, root: tk.Tk, initial_source: str | None):
        self.root = root
        self.root.title("Binary Morphology Editor")
        self.root.geometry("1400x850")
        self.root.minsize(900, 500)

        self.source_bgr: np.ndarray | None = None
        self.tk_original: ImageTk.PhotoImage | None = None
        self.tk_result: ImageTk.PhotoImage | None = None
        self._original_pil: Image.Image | None = None
        self._result01: np.ndarray | None = None

        self._build_layout()
        self._build_controls()
        self.root.update_idletasks()  # realize widget sizes before any image is rendered

        if initial_source:
            self._load_source(initial_source)

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Load Image...", command=self._on_load).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save Result...", command=self._on_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset Steps", command=self._on_reset).pack(side=tk.LEFT, padx=2)

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        sidebar_container = ttk.Frame(body, width=340)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_container.pack_propagate(False)

        canvas = tk.Canvas(sidebar_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_container, orient=tk.VERTICAL, command=canvas.yview)
        self.sidebar = ttk.Frame(canvas)
        self.sidebar.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.sidebar, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        preview_area = ttk.Frame(body, padding=6)
        preview_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        preview_area.columnconfigure(0, weight=1, uniform="pane")
        preview_area.columnconfigure(1, weight=1, uniform="pane")
        preview_area.rowconfigure(0, weight=1)

        original_frame = ttk.LabelFrame(preview_area, text="Original")
        original_frame.grid(row=0, column=0, sticky="nsew", padx=4)
        original_frame.pack_propagate(False)
        self.original_label = ttk.Label(original_frame, anchor="center")
        self.original_label.pack(fill=tk.BOTH, expand=True)
        self.original_label.bind("<Configure>", lambda _e: self._render_original())

        result_frame = ttk.LabelFrame(preview_area, text="Result (live)")
        result_frame.grid(row=0, column=1, sticky="nsew", padx=4)
        result_frame.pack_propagate(False)
        self.result_label = ttk.Label(result_frame, anchor="center")
        self.result_label.pack(fill=tk.BOTH, expand=True)
        self.result_label.bind("<Configure>", lambda _e: self._render_result())

    # -------------------------------------------------------------- controls
    def _build_controls(self):
        self.steps: list[MorphStep] = []

        threshold_frame = ttk.LabelFrame(self.sidebar, text="Binarize (threshold)")
        threshold_frame.pack(fill=tk.X, padx=6, pady=6)
        self.threshold_var = tk.IntVar(value=128)
        ttk.Scale(
            threshold_frame, from_=0, to=255, variable=self.threshold_var,
            command=lambda _v: self._on_change(),
        ).pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(threshold_frame, textvariable=self.threshold_var).pack(anchor="w", padx=6)

        self._add_kernel_step(
            "erosion", "Erosion", default_kernel=3, default_iter=1,
            apply_fn=lambda img, kernel, it: cv2.erode(img, kernel, iterations=it),
        )
        self._add_kernel_step(
            "dilation", "Dilation", default_kernel=3, default_iter=1,
            apply_fn=lambda img, kernel, it: cv2.dilate(img, kernel, iterations=it),
        )
        self._add_kernel_step(
            "opening", "Opening (erode -> dilate)", default_kernel=3, default_iter=1,
            apply_fn=lambda img, kernel, it: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it),
        )
        self._add_kernel_step(
            "closing", "Closing (dilate -> erode)", default_kernel=3, default_iter=1,
            apply_fn=lambda img, kernel, it: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=it),
        )

        self._add_hitmiss_step()
        self._add_boundary_step()
        self._add_skeleton_step()
        self._add_prune_step()

    def _add_kernel_step(self, key, title, default_kernel, default_iter, apply_fn):
        frame = ttk.LabelFrame(self.sidebar, text=title)
        frame.pack(fill=tk.X, padx=6, pady=6)

        enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable", variable=enabled_var, command=self._on_change).pack(anchor="w", padx=6)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Kernel size").pack(side=tk.LEFT)
        kernel_var = tk.IntVar(value=default_kernel)
        ttk.Spinbox(
            row, from_=1, to=31, increment=2, width=5, textvariable=kernel_var,
            command=self._on_change,
        ).pack(side=tk.LEFT, padx=6)

        shape_var = tk.StringVar(value="Rectangle")
        ttk.Combobox(
            row, textvariable=shape_var, values=list(SHAPES), width=10, state="readonly",
        ).pack(side=tk.LEFT, padx=6)
        shape_var.trace_add("write", lambda *_: self._on_change())

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row2, text="Iterations").pack(side=tk.LEFT)
        iter_var = tk.IntVar(value=default_iter)
        ttk.Spinbox(
            row2, from_=1, to=20, width=5, textvariable=iter_var, command=self._on_change,
        ).pack(side=tk.LEFT, padx=6)

        def run(img01):
            k = max(1, kernel_var.get() | 1)
            kernel = cv2.getStructuringElement(SHAPES[shape_var.get()], (k, k))
            return apply_fn(img01, kernel, max(1, iter_var.get()))

        self.steps.append(MorphStep(key, enabled_var, run))

    def _add_hitmiss_step(self):
        frame = ttk.LabelFrame(self.sidebar, text="Hit-or-miss")
        frame.pack(fill=tk.X, padx=6, pady=6)

        enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable", variable=enabled_var, command=self._on_change).pack(anchor="w", padx=6)

        preset_var = tk.StringVar(value="Isolated points")
        ttk.Combobox(
            frame, textvariable=preset_var, values=list(HITMISS_PRESETS) + ["Custom"],
            width=20, state="readonly",
        ).pack(fill=tk.X, padx=6, pady=2)
        preset_var.trace_add("write", lambda *_: self._on_change())

        ttk.Label(frame, text="Custom 3x3 kernel (rows of -1/0/1, space-separated):").pack(
            anchor="w", padx=6
        )
        custom_text = tk.Text(frame, height=3, width=20)
        custom_text.insert("1.0", "-1 -1 -1\n-1  1 -1\n-1 -1 -1")
        custom_text.pack(padx=6, pady=2)
        custom_text.bind("<KeyRelease>", lambda _e: self._on_change())

        def run(img01):
            if preset_var.get() == "Custom":
                try:
                    rows = [
                        [int(v) for v in line.split()]
                        for line in custom_text.get("1.0", tk.END).strip().splitlines()
                    ]
                    kernel = np.array(rows, dtype=np.int8)
                except ValueError:
                    return img01
            else:
                kernel = HITMISS_PRESETS[preset_var.get()]
            return cv2.morphologyEx(img01, cv2.MORPH_HITMISS, kernel)

        self.steps.append(MorphStep("hitmiss", enabled_var, run))

    def _add_boundary_step(self):
        frame = ttk.LabelFrame(self.sidebar, text="Boundary extraction (A - erosion(A))")
        frame.pack(fill=tk.X, padx=6, pady=6)

        enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable", variable=enabled_var, command=self._on_change).pack(anchor="w", padx=6)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Kernel size").pack(side=tk.LEFT)
        kernel_var = tk.IntVar(value=3)
        ttk.Spinbox(
            row, from_=1, to=31, increment=2, width=5, textvariable=kernel_var, command=self._on_change,
        ).pack(side=tk.LEFT, padx=6)

        def run(img01):
            k = max(1, kernel_var.get() | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            eroded = cv2.erode(img01, kernel, iterations=1)
            return np.clip(img01.astype(np.int16) - eroded.astype(np.int16), 0, 1).astype(np.uint8)

        self.steps.append(MorphStep("boundary", enabled_var, run))

    def _add_skeleton_step(self):
        frame = ttk.LabelFrame(self.sidebar, text="Skeletonize / thin")
        frame.pack(fill=tk.X, padx=6, pady=6)

        enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable", variable=enabled_var, command=self._on_change).pack(anchor="w", padx=6)

        method_var = tk.StringVar(value="zhang")
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Method").pack(side=tk.LEFT)
        ttk.Combobox(
            row, textvariable=method_var, values=["zhang", "lee", "thin"], width=10, state="readonly",
        ).pack(side=tk.LEFT, padx=6)
        method_var.trace_add("write", lambda *_: self._on_change())

        def run(img01):
            mask = img01.astype(bool)
            if method_var.get() == "thin":
                skeleton = thin(mask)
            else:
                skeleton = skeletonize(mask, method=method_var.get())
            return skeleton.astype(np.uint8)

        self.steps.append(MorphStep("skeleton", enabled_var, run))

    def _add_prune_step(self):
        frame = ttk.LabelFrame(self.sidebar, text="Prune (remove spurs, run after skeletonize)")
        frame.pack(fill=tk.X, padx=6, pady=6)

        enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Enable", variable=enabled_var, command=self._on_change).pack(anchor="w", padx=6)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="Spur length (iterations)").pack(side=tk.LEFT)
        iter_var = tk.IntVar(value=5)
        ttk.Spinbox(
            row, from_=1, to=50, width=5, textvariable=iter_var, command=self._on_change,
        ).pack(side=tk.LEFT, padx=6)

        def run(img01):
            return prune(img01, iter_var.get())

        self.steps.append(MorphStep("prune", enabled_var, run))

    # ---------------------------------------------------------------- events
    def _on_load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self._load_source(path)

    def _load_source(self, source: str):
        try:
            self.source_bgr = load_image_bgr(source)
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
            messagebox.showerror("Failed to load image", str(exc))
            return
        self._show_original()
        self._on_change()

    def _on_reset(self):
        for step in self.steps:
            step.enabled_var.set(False)
        self.threshold_var.set(128)
        self._on_change()

    def _on_save(self):
        result01 = self._compute_result()
        if result01 is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        Image.fromarray(result01 * 255).save(path)

    def _on_change(self):
        result01 = self._compute_result()
        if result01 is None:
            return
        self._show_result(result01)

    # ------------------------------------------------------------- pipeline
    def _compute_result(self):
        if self.source_bgr is None:
            return None
        gray = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_var.get(), 1, cv2.THRESH_BINARY)
        result = binary.astype(np.uint8)
        for step in self.steps:
            if step.enabled_var.get():
                result = step.build_controls(result)
        return result

    # --------------------------------------------------------------- render
    @staticmethod
    def _fit_to_widget(widget: tk.Widget, pil_image: Image.Image) -> Image.Image:
        width = widget.winfo_width()
        height = widget.winfo_height()
        if width < 10 or height < 10:
            return pil_image
        scale = min(width / pil_image.width, height / pil_image.height)
        new_size = (max(1, int(pil_image.width * scale)), max(1, int(pil_image.height * scale)))
        return pil_image.resize(new_size, Image.LANCZOS)

    def _show_original(self):
        rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
        self._original_pil = Image.fromarray(rgb)
        self._render_original()

    def _render_original(self):
        if self._original_pil is None:
            return
        fitted = self._fit_to_widget(self.original_label, self._original_pil)
        self.tk_original = ImageTk.PhotoImage(fitted)
        self.original_label.configure(image=self.tk_original)

    def _show_result(self, result01: np.ndarray):
        self._result01 = result01
        self._render_result()

    def _render_result(self):
        if self._result01 is None:
            return
        pil_image = Image.fromarray(self._result01 * 255)
        fitted = self._fit_to_widget(self.result_label, pil_image)
        self.tk_result = ImageTk.PhotoImage(fitted)
        self.result_label.configure(image=self.tk_result)


def main():
    initial_source = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    ImageEditorApp(root, initial_source)
    root.mainloop()


if __name__ == "__main__":
    main()
