"""Tkinter UI for comparing own DCT and Haar DWT image compression."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from algorithms import DCTResult, DWTResult, compress_dct, compress_dwt
from image_utils import array_to_image, load_grayscale_image
from metrics import coefficient_compression_ratio, format_psnr, psnr

EXPERIMENT_LEVELS = (50, 20, 5)
PREVIEW_SIZE = (340, 290)


@dataclass(frozen=True)
class ComparisonResult:
    keep_percent: int
    dct: DCTResult
    dwt: DWTResult
    dct_psnr: float
    dwt_psnr: float


def run_comparison(image: np.ndarray, keep_percent: int) -> ComparisonResult:
    keep_ratio = keep_percent / 100.0
    dct_result = compress_dct(image, keep_ratio)
    dwt_result = compress_dwt(image, keep_ratio, levels=3)
    return ComparisonResult(
        keep_percent=keep_percent,
        dct=dct_result,
        dwt=dwt_result,
        dct_psnr=psnr(image, dct_result.reconstructed),
        dwt_psnr=psnr(image, dwt_result.reconstructed),
    )


class CompressionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DCT vs DWT — Multimedia Lab 3")
        self.geometry("1360x900")
        self.minsize(980, 680)
        self.configure(bg="#080d1b")

        self.original: np.ndarray | None = None
        self.current_result: ComparisonResult | None = None
        self.current_path: Path | None = None
        self.preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self.keep_percent = tk.IntVar(value=20)
        self.status_text = tk.StringVar(value="Open an image to begin")
        self._busy = False
        self._job_queue: queue.Queue[tuple[str, object, object | None]] = queue.Queue()

        self._configure_styles()
        self._build_ui()
        self.after(50, self._poll_job_queue)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#080d1b")
        style.configure("Topbar.TFrame", background="#0d1427")
        style.configure("Panel.TFrame", background="#121a30")
        style.configure("MetricCard.TFrame", background="#18223c")
        style.configure(
            "Title.TLabel",
            background="#0d1427",
            foreground="#f5f7ff",
            font=("Arial", 22, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#0d1427",
            foreground="#93a4c7",
            font=("Arial", 10),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#121a30",
            foreground="#f5f7ff",
            font=("Arial", 13, "bold"),
        )
        style.configure(
            "Metric.TLabel",
            background="#121a30",
            foreground="#b9c5df",
            font=("Arial", 9),
        )
        style.configure(
            "Eyebrow.TLabel",
            background="#0d1427",
            foreground="#8fa2c9",
            font=("Arial", 8, "bold"),
        )
        style.configure(
            "ChipTitle.TLabel",
            background="#18223c",
            foreground="#8093bc",
            font=("Arial", 7, "bold"),
        )
        style.configure(
            "ChipValue.TLabel",
            background="#18223c",
            foreground="#f5f7ff",
            font=("Arial", 9, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#0d1427",
            foreground="#9aadd2",
            font=("Arial", 9),
        )
        style.configure(
            "Primary.TButton",
            font=("Arial", 10, "bold"),
            padding=(14, 8),
            background="#7c5cff",
            foreground="white",
        )
        style.map("Primary.TButton", background=[("active", "#947aff")])
        style.configure(
            "Secondary.TButton",
            font=("Arial", 10),
            padding=(12, 8),
            background="#1d2947",
            foreground="#e8edff",
        )
        style.map("Secondary.TButton", background=[("active", "#303d61")])
        style.configure(
            "Treeview",
            background="#121a30",
            fieldbackground="#121a30",
            foreground="#e9edff",
            rowheight=26,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#1d2947",
            foreground="#f5f7ff",
            font=("Arial", 9, "bold"),
        )

    def _build_ui(self) -> None:
        shell = ttk.Frame(self, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        self.content_canvas = tk.Canvas(
            shell, background="#080d1b", highlightthickness=0, borderwidth=0
        )
        scrollbar = ttk.Scrollbar(
            shell, orient="vertical", command=self.content_canvas.yview
        )
        self.content_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.content_canvas.pack(side="left", fill="both", expand=True)

        root = ttk.Frame(self.content_canvas, style="App.TFrame", padding=(18, 14, 18, 12))
        self._canvas_window = self.content_canvas.create_window(
            (0, 0), window=root, anchor="nw"
        )
        root.bind(
            "<Configure>",
            lambda _event: self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all")
            ),
        )
        self.content_canvas.bind(
            "<Configure>",
            lambda event: self.content_canvas.itemconfigure(
                self._canvas_window, width=event.width
            ),
        )
        self.bind_all("<MouseWheel>", self._on_mousewheel)

        header = ttk.Frame(root, style="Topbar.TFrame", padding=(16, 12))
        header.pack(fill="x", pady=(0, 10))

        brand = tk.Canvas(
            header, width=42, height=42, bg="#0d1427", highlightthickness=0
        )
        brand.pack(side="left", padx=(0, 12))
        brand.create_oval(2, 2, 40, 40, fill="#7c5cff", outline="")
        brand.create_polygon(16, 11, 32, 21, 16, 31, fill="#ffffff", outline="")

        header_copy = ttk.Frame(header, style="Topbar.TFrame")
        header_copy.pack(side="left", fill="x", expand=True)
        ttk.Label(header_copy, text="TRANSFORM LAB · LAB 3", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(header_copy, text="DCT vs DWT", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="DCT 8×8   •   HAAR DWT   •   PSNR",
            style="Eyebrow.TLabel",
        ).pack(side="right", padx=(12, 2))

        controls = ttk.Frame(root, style="Panel.TFrame", padding=(14, 11))
        controls.pack(fill="x", pady=(0, 10))

        self.open_button = ttk.Button(
            controls,
            text="Open image",
            style="Primary.TButton",
            command=self.open_image,
        )
        self.open_button.pack(side="left")

        ttk.Label(
            controls,
            text="Keep coefficients:",
            style="Metric.TLabel",
        ).pack(side="left", padx=(20, 8))
        self.level_scale = ttk.Scale(
            controls,
            from_=1,
            to=100,
            variable=self.keep_percent,
            command=self._on_scale,
            length=230,
        )
        self.level_scale.pack(side="left")
        self.level_label = ttk.Label(
            controls,
            text="20%",
            style="PanelTitle.TLabel",
            width=5,
        )
        self.level_label.pack(side="left", padx=(8, 12))

        self.process_button = ttk.Button(
            controls,
            text="Apply",
            style="Secondary.TButton",
            command=self.process_current_level,
            state="disabled",
        )
        self.process_button.pack(side="left", padx=4)
        self.experiment_button = ttk.Button(
            controls,
            text="Experiment 50% / 20% / 5%",
            style="Secondary.TButton",
            command=self.run_experiment,
            state="disabled",
        )
        self.experiment_button.pack(side="left", padx=4)

        image_row = ttk.Frame(root, style="App.TFrame")
        image_row.pack(fill="x")
        image_row.columnconfigure((0, 1, 2), weight=1, uniform="preview")
        image_row.rowconfigure(0, weight=1)

        self.image_labels: dict[str, ttk.Label] = {}
        self.metric_labels: dict[str, ttk.Label] = {}
        self.metric_values: dict[str, dict[str, ttk.Label]] = {}
        for column, (key, title) in enumerate(
            (("original", "Original"), ("dct", "DCT 8×8"), ("dwt", "Haar DWT"))
        ):
            panel = ttk.Frame(image_row, style="Panel.TFrame", padding=12)
            panel.grid(row=0, column=column, sticky="nsew", padx=5)
            ttk.Label(panel, text=title, style="PanelTitle.TLabel").pack(anchor="w")

            preview = tk.Frame(panel, height=300, bg="#070b16")
            preview.pack(fill="x", pady=(9, 9))
            preview.pack_propagate(False)
            image_label = ttk.Label(
                preview,
                text="No image",
                anchor="center",
                background="#070b16",
                foreground="#647295",
            )
            image_label.pack(fill="both", expand=True)

            metric_strip = ttk.Frame(panel, style="Panel.TFrame")
            metric_strip.pack(fill="x")
            metric_strip.columnconfigure((0, 1, 2), weight=1, uniform="metrics")
            self.metric_values[key] = {}
            card_defs = (
                ("primary", "IMAGE" if key == "original" else "PSNR"),
                ("secondary", "MODE" if key == "original" else "RETAINED"),
                ("tertiary", "SIZE" if key == "original" else "RATIO"),
            )
            for metric_column, (metric_key, caption) in enumerate(card_defs):
                metric_card = ttk.Frame(metric_strip, style="MetricCard.TFrame", padding=(8, 6))
                metric_card.grid(
                    row=0, column=metric_column, sticky="ew", padx=(0 if metric_column == 0 else 3, 0)
                )
                ttk.Label(metric_card, text=caption, style="ChipTitle.TLabel").pack(anchor="w")
                value_label = ttk.Label(metric_card, text="—", style="ChipValue.TLabel")
                value_label.pack(anchor="w", pady=(2, 0))
                self.metric_values[key][metric_key] = value_label

            metric_label = ttk.Label(panel, text="", style="Metric.TLabel")
            self.image_labels[key] = image_label
            self.metric_labels[key] = metric_label

        lower = ttk.Frame(root, style="Panel.TFrame", padding=12)
        lower.pack(fill="x", pady=(10, 0))
        lower.columnconfigure(0, weight=1)

        ttk.Label(
            lower,
            text="Experiment results",
            style="PanelTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        columns = ("level", "dct_count", "dct_psnr", "dwt_count", "dwt_psnr")
        self.results_table = ttk.Treeview(
            lower,
            columns=columns,
            show="headings",
            height=3,
        )
        headings = {
            "level": "Retained",
            "dct_count": "DCT coeff.",
            "dct_psnr": "DCT PSNR",
            "dwt_count": "DWT coeff.",
            "dwt_psnr": "DWT PSNR",
        }
        for name in columns:
            self.results_table.heading(name, text=headings[name])
            self.results_table.column(name, anchor="center", width=155)
        self.results_table.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.save_dct_button = ttk.Button(
            lower,
            text="Save DCT",
            style="Secondary.TButton",
            command=lambda: self.save_result("dct"),
            state="disabled",
        )
        self.save_dct_button.grid(row=0, column=1, padx=4)
        self.save_dwt_button = ttk.Button(
            lower,
            text="Save DWT",
            style="Secondary.TButton",
            command=lambda: self.save_result("dwt"),
            state="disabled",
        )
        self.save_dwt_button.grid(row=0, column=2, padx=4)

        status_bar = ttk.Frame(root, style="Topbar.TFrame", padding=(12, 8))
        status_bar.pack(fill="x", pady=(10, 0))
        ttk.Label(status_bar, text="●", foreground="#46d7a2", background="#0d1427").pack(side="left")
        ttk.Label(
            status_bar, textvariable=self.status_text, style="Status.TLabel"
        ).pack(side="left", padx=(7, 0))

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.content_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_scale(self, raw_value: str) -> None:
        rounded = max(1, min(100, int(round(float(raw_value)))))
        self.keep_percent.set(rounded)
        self.level_label.configure(text=f"{rounded}%")

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.original = load_grayscale_image(path)
        except Exception as exc:
            messagebox.showerror("Open error", str(exc))
            return

        self.current_path = Path(path)
        self.current_result = None
        self._show_image("original", self.original)
        self.image_labels["dct"].configure(image="", text="Press Apply")
        self.image_labels["dwt"].configure(image="", text="Press Apply")
        short_name = self.current_path.name
        if len(short_name) > 18:
            short_name = f"{short_name[:14]}..."
        self._set_metrics("original", short_name, "Grayscale", "512×512")
        self._set_metrics("dct", "—", "—", "—")
        self._set_metrics("dwt", "—", "—", "—")
        self.process_button.configure(state="normal")
        self.experiment_button.configure(state="normal")
        self.save_dct_button.configure(state="disabled")
        self.save_dwt_button.configure(state="disabled")
        self.status_text.set("Image is ready for processing")

    def process_current_level(self) -> None:
        if self.original is None or self._busy:
            return
        self._start_job(
            lambda: run_comparison(self.original.copy(), self.keep_percent.get()),
            self._finish_single_result,
            f"Calculating DCT and DWT for {self.keep_percent.get()}% of coefficients…",
        )

    def run_experiment(self) -> None:
        if self.original is None or self._busy:
            return

        source = self.original.copy()

        def calculate() -> list[ComparisonResult]:
            return [run_comparison(source, level) for level in EXPERIMENT_LEVELS]

        self._start_job(
            calculate,
            self._finish_experiment,
            "Running three experiments…",
        )

    def _start_job(self, work, on_success, status: str) -> None:
        self._busy = True
        self.status_text.set(status)
        self._set_controls_enabled(False)

        def runner() -> None:
            try:
                result = work()
            except Exception as exc:
                self._job_queue.put(("error", exc, None))
                return
            self._job_queue.put(("success", result, on_success))

        threading.Thread(target=runner, daemon=True).start()

    def _poll_job_queue(self) -> None:
        """Move completed worker results to Tk's main thread safely."""

        try:
            while True:
                state, payload, callback = self._job_queue.get_nowait()
                if state == "error":
                    self._finish_error(payload)  # type: ignore[arg-type]
                elif callback is not None:
                    callback(payload)
        except queue.Empty:
            pass
        finally:
            self.after(50, self._poll_job_queue)

    def _finish_single_result(self, result: ComparisonResult) -> None:
        self.current_result = result
        self._display_comparison(result)
        self._upsert_table_row(result)
        self.status_text.set(
            f"Done: retained {result.keep_percent}% of coefficients"
        )
        self._finish_job()

    def _finish_experiment(self, results: list[ComparisonResult]) -> None:
        for row in self.results_table.get_children():
            self.results_table.delete(row)
        for result in results:
            self._upsert_table_row(result)

        selected_percent = self.keep_percent.get()
        self.current_result = min(
            results, key=lambda item: abs(item.keep_percent - selected_percent)
        )
        self.keep_percent.set(self.current_result.keep_percent)
        self.level_label.configure(text=f"{self.current_result.keep_percent}%")
        self._display_comparison(self.current_result)
        self.status_text.set("Experiment completed: 50%, 20% and 5%")
        self._finish_job()

    def _finish_error(self, exc: Exception) -> None:
        messagebox.showerror("Processing error", str(exc))
        self.status_text.set("The calculation could not be completed")
        self._finish_job()

    def _finish_job(self) -> None:
        self._busy = False
        self._set_controls_enabled(True)
        if self.current_result is not None:
            self.save_dct_button.configure(state="normal")
            self.save_dwt_button.configure(state="normal")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.open_button.configure(state=state)
        self.process_button.configure(state=state if self.original is not None else "disabled")
        self.experiment_button.configure(state=state if self.original is not None else "disabled")
        self.level_scale.configure(state=state)

    def _display_comparison(self, result: ComparisonResult) -> None:
        self._show_image("dct", result.dct.reconstructed)
        self._show_image("dwt", result.dwt.reconstructed)

        dct_ratio = coefficient_compression_ratio(
            result.dct.total_count, result.dct.retained_count
        )
        dwt_ratio = coefficient_compression_ratio(
            result.dwt.total_count, result.dwt.retained_count
        )
        self._set_metrics(
            "dct",
            format_psnr(result.dct_psnr),
            f"{result.keep_percent}%",
            f"{dct_ratio:.2f}:1",
        )
        self._set_metrics(
            "dwt",
            format_psnr(result.dwt_psnr),
            f"{result.keep_percent}%",
            f"{dwt_ratio:.2f}:1",
        )

    def _set_metrics(self, key: str, primary: str, secondary: str, tertiary: str) -> None:
        values = self.metric_values[key]
        values["primary"].configure(text=primary)
        values["secondary"].configure(text=secondary)
        values["tertiary"].configure(text=tertiary)

    def _show_image(self, key: str, array: np.ndarray) -> None:
        image = array_to_image(array)
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.preview_refs[key] = photo
        self.image_labels[key].configure(image=photo, text="")

    def _upsert_table_row(self, result: ComparisonResult) -> None:
        item_id = f"level-{result.keep_percent}"
        values = (
            f"{result.keep_percent}%",
            f"{result.dct.retained_count:,}".replace(",", " "),
            format_psnr(result.dct_psnr),
            f"{result.dwt.retained_count:,}".replace(",", " "),
            format_psnr(result.dwt_psnr),
        )
        if self.results_table.exists(item_id):
            self.results_table.item(item_id, values=values)
        else:
            self.results_table.insert("", "end", iid=item_id, values=values)

    def save_result(self, method: str) -> None:
        if self.current_result is None:
            return
        default_name = (
            f"{self.current_path.stem if self.current_path else 'image'}_"
            f"{method}_{self.current_result.keep_percent}percent.png"
        )
        path = filedialog.asksaveasfilename(
            title=f"Save {method.upper()} result",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not path:
            return

        array = (
            self.current_result.dct.reconstructed
            if method == "dct"
            else self.current_result.dwt.reconstructed
        )
        array_to_image(array).save(path)
        self.status_text.set(f"Result saved: {path}")


def self_test() -> None:
    """Quick headless smoke test used in environments without a display."""

    axis = np.linspace(0, 255, 64)
    image = np.add.outer(axis, axis) / 2
    result = run_comparison(image, 20)
    assert result.dct.reconstructed.shape == image.shape
    assert result.dwt.reconstructed.shape == image.shape
    assert np.isfinite(result.dct_psnr)
    assert np.isfinite(result.dwt_psnr)
    print(
        "Self-test passed:",
        f"DCT={result.dct_psnr:.2f} dB,",
        f"DWT={result.dwt_psnr:.2f} dB",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a headless algorithm smoke test",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    app = CompressionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
