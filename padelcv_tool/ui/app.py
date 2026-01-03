"""PySide6 desktop application for padel stroke inference."""

from __future__ import annotations

import html
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import CLIP_NUM_FRAMES, CLIP_IMAGE_SIZE, ensure_artifact_dirs
from ..data import ClipSelection, extract_clip_frames, frames_to_tensor, load_class_names
from ..models import PredictionResult, StrokeInferenceEngine
from ..pose import PoseAnimation, PoseEstimator
from .widgets import ClipTimeline, VideoFrameView


@dataclass
class VideoMetadata:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


class VideoLoader:
    def load(self, path: Path) -> VideoMetadata:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        meta = VideoMetadata(
            path=path,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        return meta

    def read_frame(self, meta: VideoMetadata, index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(meta.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Unable to read frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class InferenceWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        engine: StrokeInferenceEngine,
        pose_estimator: PoseEstimator,
        selection: ClipSelection,
        model_key: str,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.pose_estimator = pose_estimator
        self.selection = selection
        self.model_key = model_key

    @QtCore.Slot()
    def run(self) -> None:
        try:
            frames = extract_clip_frames(
                self.selection.video_path,
                self.selection.start_frame,
                self.selection.end_frame,
                self.selection.bbox_norm,
                num_frames=CLIP_NUM_FRAMES,
                output_size=CLIP_IMAGE_SIZE,
            )
            clip_tensor = frames_to_tensor(frames)
            result = self.engine.predict(self.model_key, clip_tensor)
            pose_animation = self.pose_estimator.render_animation(frames)
            self.finished.emit(
                {
                    "prediction": result,
                    "pose": pose_animation,
                    "frame_count": len(frames),
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class StrokeInferenceWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_artifact_dirs()
        self.setWindowTitle("Padel Stroke Inference")
        self.resize(1400, 800)

        self.video_loader = VideoLoader()
        self.video_meta: Optional[VideoMetadata] = None
        self.current_frame_idx = 0
        self.selection_bbox = (0.1, 0.1, 0.8, 0.8)
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_last_frame = -1
        self._ignore_spin_signals = False
        self._ignore_timeline_signals = False

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance_playback)
        self._playback_range = (0, 0)
        self._playback_selection_only = False
        self._playback_should_loop = False

        self.pose_timer = QtCore.QTimer(self)
        self.pose_timer.timeout.connect(self._advance_pose_frame)
        self._pose_pixmaps: list[QtGui.QPixmap] = []
        self._pose_frame_index = 0

        class_names = load_class_names()
        self.engine = StrokeInferenceEngine(class_names)
        self.pose_estimator = PoseEstimator()

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)

        left_panel = QtWidgets.QVBoxLayout()
        self.frame_view = VideoFrameView()
        self.frame_view.set_bbox(self.selection_bbox)
        self.frame_view.bboxChanged.connect(self._on_bbox_changed)
        left_panel.addWidget(self.frame_view, stretch=5)

        controls = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("Load Video…")
        self.load_button.clicked.connect(self._load_video)
        controls.addWidget(self.load_button)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button)
        self.preview_button = QtWidgets.QPushButton("Preview Selection")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._preview_selection)
        controls.addWidget(self.preview_button)
        self.mark_in_button = QtWidgets.QPushButton("Mark In")
        self.mark_in_button.setEnabled(False)
        self.mark_in_button.clicked.connect(self._mark_in)
        controls.addWidget(self.mark_in_button)
        self.mark_out_button = QtWidgets.QPushButton("Mark Out")
        self.mark_out_button.setEnabled(False)
        self.mark_out_button.clicked.connect(self._mark_out)
        controls.addWidget(self.mark_out_button)
        left_panel.addLayout(controls)

        self.timeline = ClipTimeline()
        self.timeline.setEnabled(False)
        self.timeline.rangeChanged.connect(self._on_timeline_range_changed)
        self.timeline.currentFrameChanged.connect(self._on_timeline_frame_changed)
        left_panel.addWidget(self.timeline)

        selection_layout = QtWidgets.QFormLayout()
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setEnabled(False)
        self.start_spin.valueChanged.connect(self._ensure_frame_order)
        selection_layout.addRow("Start frame", self.start_spin)
        self.end_spin = QtWidgets.QSpinBox()
        self.end_spin.setEnabled(False)
        self.end_spin.valueChanged.connect(self._ensure_frame_order)
        selection_layout.addRow("End frame", self.end_spin)
        self.bbox_label = QtWidgets.QLabel("BBox: (—)")
        selection_layout.addRow("Bounding box", self.bbox_label)
        self.loop_checkbox = QtWidgets.QCheckBox("Loop selection")
        self.loop_checkbox.setEnabled(False)
        self.loop_checkbox.stateChanged.connect(self._on_loop_toggled)
        selection_layout.addRow("Playback", self.loop_checkbox)
        left_panel.addLayout(selection_layout)

        layout.addLayout(left_panel, stretch=3)

        right_container = QtWidgets.QWidget()
        right_container.setMaximumWidth(450)
        right_panel = QtWidgets.QVBoxLayout(right_container)
        self.meta_label = QtWidgets.QLabel("Load a video to begin.")
        self.meta_label.setWordWrap(True)
        right_panel.addWidget(self.meta_label)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["i3d", "timesformer"])
        right_panel.addWidget(self.model_combo)

        self.run_button = QtWidgets.QPushButton("Run Inference")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run_inference)
        right_panel.addWidget(self.run_button)

        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(180)
        self.results_text.setStyleSheet(
            "QTextEdit { font-family: 'Menlo', 'Consolas', monospace; background-color: #0f0f0f; border: 1px solid #333; color: #ddd; }"
        )
        right_panel.addWidget(self.results_text, stretch=2)

        self.pose_label = QtWidgets.QLabel("Pose preview will appear here.")
        self.pose_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pose_label.setStyleSheet(
            "background-color: #050505; color: #777; border: 1px solid #333;"
        )
        self.pose_label.setMinimumHeight(360)
        self.pose_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
        )
        right_panel.addWidget(self.pose_label, stretch=3)

        layout.addWidget(right_container, stretch=2)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _load_video(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", str(Path.home()), "Video Files (*.mp4 *.mov *.mkv)"
        )
        if not file_path:
            return
        path = Path(file_path)
        self._stop_playback()
        self._release_capture()
        try:
            meta = self.video_loader.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Unable to load video", str(exc))
            return
        try:
            self._open_capture(path)
        except RuntimeError as exc:
            QtWidgets.QMessageBox.critical(self, "Unable to open video", str(exc))
            self.video_meta = None
            return
        self.video_meta = meta
        self.current_frame_idx = 0
        max_frame = max(0, meta.frame_count - 1)
        self.start_spin.setEnabled(True)
        self.end_spin.setEnabled(True)
        self.start_spin.setRange(0, max_frame)
        self.end_spin.setRange(0, max_frame)
        self.start_spin.setValue(0)
        self.end_spin.setValue(min(CLIP_NUM_FRAMES - 1, max_frame))
        self.mark_in_button.setEnabled(True)
        self.mark_out_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.loop_checkbox.setEnabled(True)
        self.loop_checkbox.setChecked(False)
        self.timeline.setEnabled(True)
        self.timeline.set_duration(meta.frame_count)
        self.timeline.set_range(self.start_spin.value(), self.end_spin.value())
        self.timeline.set_current_frame(0)
        self._playback_range = self._compute_playback_range(self._playback_selection_only)
        self._update_meta_label()
        self._display_current_frame()
        self._set_pose_preview(None, "Pose preview will appear here.")

    def _update_meta_label(self) -> None:
        if not self.video_meta:
            self.meta_label.setText("Load a video to begin.")
            return
        meta = self.video_meta
        self.meta_label.setText(
            f"Video: {meta.path.name}\n"
            f"Frames: {meta.frame_count} | FPS: {meta.fps:.2f}\n"
            f"Resolution: {meta.width}x{meta.height}"
        )

    def _display_current_frame(self) -> None:
        if not self.video_meta:
            return
        try:
            frame = self._read_video_frame(self.current_frame_idx)
        except Exception as exc:
            self.statusBar().showMessage(f"Preview error: {exc}")
            return
        self.frame_view.set_frame(frame)

    def _mark_in(self) -> None:
        if not self.video_meta:
            return
        self.start_spin.setValue(self.current_frame_idx)

    def _mark_out(self) -> None:
        if not self.video_meta:
            return
        self.end_spin.setValue(self.current_frame_idx)

    def _ensure_frame_order(self) -> None:
        if self._ignore_spin_signals:
            return
        start_val = self.start_spin.value()
        end_val = self.end_spin.value()
        if end_val < start_val:
            self._ignore_spin_signals = True
            self.end_spin.setValue(start_val)
            self._ignore_spin_signals = False
            end_val = start_val
        self._sync_timeline_to_spins()

    def _on_bbox_changed(self, bbox: tuple) -> None:
        self.selection_bbox = bbox
        x, y, w, h = bbox
        self.bbox_label.setText(f"x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")

    def _on_timeline_range_changed(self, start: int, end: int) -> None:
        if self._ignore_timeline_signals:
            return
        self._ignore_spin_signals = True
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self._ignore_spin_signals = False
        self._playback_range = self._compute_playback_range(self._playback_selection_only)
        self._sync_sliders_to_spins()

    def _on_timeline_frame_changed(self, frame: int) -> None:
        if self._ignore_timeline_signals:
            return
        self.current_frame_idx = frame
        self._display_current_frame()

    def _sync_timeline_to_spins(self) -> None:
        if not self.timeline.isEnabled():
            return
        self._ignore_timeline_signals = True
        self.timeline.set_range(self.start_spin.value(), self.end_spin.value())
        self._ignore_timeline_signals = False
        self._playback_range = self._compute_playback_range(self._playback_selection_only)

    def _toggle_playback(self) -> None:
        if not self.video_meta:
            return
        if self.play_timer.isActive():
            self._stop_playback()
        else:
            self._start_playback()

    def _preview_selection(self) -> None:
        if not self.video_meta:
            return
        self._set_current_frame(self.start_spin.value())
        self._start_playback(selection_only=True, loop_override=False)

    def _set_current_frame(self, frame: int) -> None:
        if not self.video_meta:
            return
        max_frame = max(0, self.video_meta.frame_count - 1)
        frame = max(0, min(frame, max_frame))
        self.current_frame_idx = frame
        if self.timeline.isEnabled():
            self._ignore_timeline_signals = True
            self.timeline.set_current_frame(frame)
            self._ignore_timeline_signals = False
        self._display_current_frame()

    def _start_playback(
        self,
        selection_only: bool | None = None,
        loop_override: bool | None = None,
    ) -> None:
        if not self.video_meta:
            return
        if selection_only is None:
            selection_only = self.loop_checkbox.isChecked()
        if loop_override is None:
            loop_override = selection_only and self.loop_checkbox.isChecked()
        self._playback_selection_only = bool(selection_only)
        self._playback_should_loop = bool(loop_override)
        self._playback_range = self._compute_playback_range(self._playback_selection_only)
        start_idx, end_idx = self._playback_range
        if start_idx > end_idx:
            self.statusBar().showMessage("Playback range is invalid")
            self.play_button.setChecked(False)
            self.play_button.setText("Play")
            return
        current = self.current_frame_idx
        if current < start_idx or current > end_idx:
            self._set_current_frame(start_idx)
        fps = self.video_meta.fps or 30.0
        interval = max(20, int(1000 / max(fps, 1.0)))
        self.play_timer.start(interval)
        self.play_button.setChecked(True)
        self.play_button.setText("Pause")
        self.statusBar().showMessage("Playing clip…")

    def _compute_playback_range(self, selection_only: bool) -> tuple[int, int]:
        if not self.video_meta:
            return (0, 0)
        if selection_only:
            start = min(self.start_spin.value(), self.end_spin.value())
            end = max(self.start_spin.value(), self.end_spin.value())
            return start, end
        return 0, max(0, self.video_meta.frame_count - 1)

    def _advance_playback(self) -> None:
        if not self.video_meta:
            self._stop_playback()
            return
        start_idx, end_idx = self._playback_range
        current = self.current_frame_idx
        if current >= end_idx:
            if self._playback_should_loop:
                self._set_current_frame(start_idx)
            else:
                self._stop_playback()
            return
        self._set_current_frame(current + 1)

    def _stop_playback(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
        self.play_button.setChecked(False)
        self.play_button.setText("Play")

    def _on_loop_toggled(self, state: int) -> None:
        if self.play_timer.isActive():
            self._playback_should_loop = bool(state) and self._playback_selection_only

    def _set_pose_preview(
        self, animation: PoseAnimation | None, placeholder: str | None = None
    ) -> None:
        self.pose_timer.stop()
        self._pose_pixmaps = []
        if animation is None or not animation.frames:
            self.pose_label.setPixmap(QtGui.QPixmap())
            if placeholder:
                self.pose_label.setText(placeholder)
            else:
                self.pose_label.setText("Pose preview will appear here.")
            return
        self._pose_pixmaps = [self._numpy_to_pixmap(frame) for frame in animation.frames]
        self._pose_frame_index = 0
        interval = max(40, int(1000 / max(animation.fps, 1.0)))
        self.pose_timer.start(interval)
        self._apply_pose_pixmap(self._pose_pixmaps[0])

    def _advance_pose_frame(self) -> None:
        if not self._pose_pixmaps:
            self.pose_timer.stop()
            return
        self._pose_frame_index = (self._pose_frame_index + 1) % len(self._pose_pixmaps)
        self._apply_pose_pixmap(self._pose_pixmaps[self._pose_frame_index])

    def _apply_pose_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap.isNull():
            self.pose_label.setPixmap(QtGui.QPixmap())
            return
        scaled = pixmap.scaled(
            self.pose_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.pose_label.setPixmap(scaled)
        self.pose_label.setText("")

    def _open_capture(self, path: Path) -> None:
        self._release_capture()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        self._capture = cap
        self._capture_last_frame = -1

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._capture_last_frame = -1

    def _read_video_frame(self, index: int) -> np.ndarray:
        if self._capture is None or not self.video_meta:
            if not self.video_meta:
                raise RuntimeError("No video loaded")
            return self.video_loader.read_frame(self.video_meta, index)
        if self._capture_last_frame + 1 != index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self._capture.read()
        if not ok:
            frame = self.video_loader.read_frame(self.video_meta, index)
            self._capture_last_frame = index
            return frame
        self._capture_last_frame = index
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _run_inference(self) -> None:
        if not self.video_meta:
            return
        bbox = self.selection_bbox
        if bbox is None:
            QtWidgets.QMessageBox.warning(
                self, "Bounding box", "Please draw a bounding box on the video preview."
            )
            return
        selection = ClipSelection(
            video_path=self.video_meta.path,
            start_frame=self.start_spin.value(),
            end_frame=self.end_spin.value(),
            bbox_norm=bbox,
        )
        model_key = self.model_combo.currentText()
        self.run_button.setEnabled(False)
        self.results_text.setHtml("<p><i>Running inference…</i></p>")
        self.statusBar().showMessage(f"Running {model_key} inference…")
        self._set_pose_preview(None, "Generating pose preview…")

        self.worker = InferenceWorker(self.engine, self.pose_estimator, selection, model_key)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_inference_finished)
        self.worker.failed.connect(self._on_inference_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_inference_finished(self, payload: object) -> None:
        self.run_button.setEnabled(True)
        self.statusBar().showMessage("Inference completed")
        if not isinstance(payload, dict):
            return
        prediction: PredictionResult = payload["prediction"]
        pose_animation: PoseAnimation | None = payload.get("pose")
        rows = []
        for idx, (label, prob) in enumerate(prediction.probabilities):
            safe_label = html.escape(label)
            highlight = idx == 0
            indicator = "🏅" if highlight else ""
            row_style = (
                "background-color:#1e2a44;font-weight:bold;color:#f8f8f2;" if highlight else ""
            )
            rows.append(
                "<tr style='{style}'>"
                '<td style="padding:2px 8px;">{icon} {label}</td>'
                '<td style="padding:2px 8px; text-align:right;">{prob:.3f}</td>'
                "</tr>".format(
                    style=row_style,
                    icon=indicator,
                    label=safe_label,
                    prob=prob,
                )
            )
        summary_html = (
            f"<p><b>Model:</b> {html.escape(prediction.model_name)}<br>"
            f"<b>Inference:</b> {prediction.inference_ms:.1f} ms</p>"
        )
        table_html = (
            '<table style="width:100%; border-collapse:collapse;">'
            "<thead><tr>"
            '<th style="text-align:left; padding:4px 8px;">Class</th>'
            '<th style="text-align:right; padding:4px 8px;">Probability</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
        self.results_text.setHtml(summary_html + table_html)
        if pose_animation is not None:
            self._set_pose_preview(pose_animation)
        else:
            self._set_pose_preview(None, "Pose preview unavailable.")

    def _on_inference_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.statusBar().showMessage("Inference failed")
        self._set_pose_preview(None, "Pose preview unavailable.")
        QtWidgets.QMessageBox.critical(self, "Inference failed", message)

    def _numpy_to_pixmap(self, image: np.ndarray) -> QtGui.QPixmap:
        h, w, _ = image.shape
        qimage = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimage.copy())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._release_capture()
        super().closeEvent(event)


def launch_app() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = StrokeInferenceWindow()
    window.show()
    app.exec()


__all__ = ["launch_app", "StrokeInferenceWindow"]
