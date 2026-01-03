"""Reusable Qt widgets for the inference UI."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class VideoFrameView(QtWidgets.QLabel):
    bboxChanged = QtCore.Signal(tuple)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #050505; border: 1px solid #333;")
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._paint_rect = QtCore.QRectF()
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._drag_origin: Optional[QtCore.QPointF] = None

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            self._pixmap = None
            self.update()
            return
        h, w, _ = frame.shape
        image = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self._pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    def set_bbox(self, bbox: Tuple[float, float, float, float] | None) -> None:
        self._bbox = bbox
        self.update()

    def current_bbox(self) -> Tuple[float, float, float, float] | None:
        return self._bbox

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111"))
        if not self._pixmap:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        self._paint_rect = QtCore.QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(QtCore.QPointF(x, y), scaled)
        if self._bbox:
            rect = self._normalized_to_rect(self._bbox)
            pen = QtGui.QPen(QtGui.QColor("#ffcc00"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._paint_rect.contains(
            event.position()
        ):
            self._drag_origin = event.position()
            norm = self._point_to_normalized(event.position())
            self._bbox = (*norm, 0.0, 0.0)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is not None:
            start = self._point_to_normalized(self._drag_origin)
            end = self._point_to_normalized(event.position())
            x0, y0 = start
            x1, y1 = end
            x = max(0.0, min(x0, x1))
            y = max(0.0, min(y0, y1))
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            self._bbox = (x, y, max(w, 1e-3), max(h, 1e-3))
            self.bboxChanged.emit(self._bbox)
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _point_to_normalized(self, point: QtCore.QPointF) -> Tuple[float, float]:
        if self._paint_rect.width() == 0 or self._paint_rect.height() == 0:
            return 0.0, 0.0
        nx = (point.x() - self._paint_rect.left()) / self._paint_rect.width()
        ny = (point.y() - self._paint_rect.top()) / self._paint_rect.height()
        return float(np.clip(nx, 0.0, 1.0)), float(np.clip(ny, 0.0, 1.0))

    def _normalized_to_rect(self, bbox: Tuple[float, float, float, float]) -> QtCore.QRectF:
        x, y, w, h = bbox
        left = self._paint_rect.left() + x * self._paint_rect.width()
        top = self._paint_rect.top() + y * self._paint_rect.height()
        width = w * self._paint_rect.width()
        height = h * self._paint_rect.height()
        return QtCore.QRectF(left, top, width, height)


class ClipTimeline(QtWidgets.QWidget):
    rangeChanged = QtCore.Signal(int, int)
    currentFrameChanged = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(72)
        self.setMouseTracking(True)
        self._duration = 0
        self._start = 0
        self._end = 0
        self._current = 0
        self._padding = 16
        self._handle_radius = 10
        self._active_handle: Optional[str] = None
        self._handle_hit_slack = 12
        self._handle_cap_width = 26
        self._handle_cap_height = 18

    def set_duration(self, frames: int) -> None:
        frames = max(0, int(frames))
        self._duration = frames
        self._start = 0
        self._end = max(0, frames - 1)
        self._current = 0
        self.update()

    def set_range(self, start: int, end: int) -> None:
        start, end = self._clamp_range(start, end)
        if start == self._start and end == self._end:
            return
        self._start, self._end = start, end
        self.update()

    def set_current_frame(self, frame: int) -> None:
        if self._set_current_frame_value(frame):
            self.update()

    def duration(self) -> int:
        return self._duration

    def range(self) -> Tuple[int, int]:
        return self._start, self._end

    def current_frame(self) -> int:
        return self._current

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled() or self._duration <= 0:
            super().mousePressEvent(event)
            return
        pos = event.position().x()
        start_x = self._frame_to_pos(self._start)
        end_x = self._frame_to_pos(self._end)
        if abs(pos - start_x) <= self._handle_hit_slack:
            self._active_handle = "start"
        elif abs(pos - end_x) <= self._handle_hit_slack:
            self._active_handle = "end"
        elif self._track_rect().contains(event.position()):
            self._active_handle = "playhead"
            self._update_playhead_from_pos(pos, emit=True)
        else:
            self._active_handle = None
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self.isEnabled() or self._duration <= 0 or self._active_handle is None:
            super().mouseMoveEvent(event)
            return
        pos = event.position().x()
        if self._active_handle == "start":
            new_start = self._pos_to_frame(pos)
            if new_start != self._start:
                self._start = min(new_start, self._end)
                self.update()
                self.rangeChanged.emit(self._start, self._end)
            self._set_current_frame_value(self._start, emit=True)
        elif self._active_handle == "end":
            new_end = self._pos_to_frame(pos)
            if new_end != self._end:
                self._end = max(new_end, self._start)
                self.update()
                self.rangeChanged.emit(self._start, self._end)
            self._set_current_frame_value(self._end, emit=True)
        elif self._active_handle == "playhead":
            if self._update_playhead_from_pos(pos, emit=True):
                self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._active_handle is not None:
            self._active_handle = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        if self._active_handle is not None:
            self._active_handle = None
        super().leaveEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#050505"))
        track = self._track_rect()
        painter.setPen(QtGui.QPen(QtGui.QColor("#222"), 1))
        painter.setBrush(QtGui.QColor("#1a2133"))
        painter.drawRoundedRect(track, 6, 6)
        if self._duration <= 0:
            return
        start_x = self._frame_to_pos(self._start)
        end_x = self._frame_to_pos(self._end)
        selection_rect = QtCore.QRectF(start_x, track.top(), end_x - start_x, track.height())
        painter.setBrush(QtGui.QColor("#315dff"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#5c7cff"), 1))
        painter.drawRoundedRect(selection_rect, 6, 6)
        cap_pen = QtGui.QPen(QtGui.QColor("#f1a208"), 1)
        cap_brush = QtGui.QBrush(QtGui.QColor("#f7c948"))
        painter.setPen(cap_pen)
        painter.setBrush(cap_brush)
        start_cap = QtCore.QRectF(
            start_x - self._handle_cap_width / 2,
            track.top() - self._handle_cap_height - 8,
            self._handle_cap_width,
            self._handle_cap_height,
        )
        end_cap = QtCore.QRectF(
            end_x - self._handle_cap_width / 2,
            track.bottom() + 8,
            self._handle_cap_width,
            self._handle_cap_height,
        )
        painter.drawRoundedRect(start_cap, 4, 4)
        painter.drawRoundedRect(end_cap, 4, 4)
        painter.drawLine(
            QtCore.QPointF(start_x, track.top()), QtCore.QPointF(start_x, start_cap.bottom())
        )
        painter.drawLine(
            QtCore.QPointF(end_x, track.bottom()), QtCore.QPointF(end_x, end_cap.top())
        )
        label_pen = QtGui.QPen(QtGui.QColor("#1c1600"))
        painter.setPen(label_pen)
        label_font = painter.font()
        label_font.setPointSizeF(max(7.0, label_font.pointSizeF() - 1))
        label_font.setWeight(QtGui.QFont.Weight.DemiBold)
        painter.setFont(label_font)
        painter.drawText(start_cap, QtCore.Qt.AlignmentFlag.AlignCenter, "IN")
        painter.drawText(end_cap, QtCore.Qt.AlignmentFlag.AlignCenter, "OUT")
        playhead_x = self._frame_to_pos(self._current)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        painter.drawLine(
            QtCore.QPointF(playhead_x, track.top() - 8),
            QtCore.QPointF(playhead_x, track.bottom() + 8),
        )

    def _track_rect(self) -> QtCore.QRectF:
        rect = self.rect()
        return QtCore.QRectF(
            rect.left() + self._padding,
            rect.center().y() - 12,
            max(10.0, rect.width() - 2 * self._padding),
            24,
        )

    def _frame_to_pos(self, frame: int) -> float:
        track = self._track_rect()
        if self._duration <= 1:
            return track.left()
        ratio = frame / float(self._duration - 1)
        return track.left() + ratio * track.width()

    def _pos_to_frame(self, pos: float) -> int:
        track = self._track_rect()
        if track.width() <= 0:
            return 0
        ratio = (pos - track.left()) / track.width()
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(ratio * max(0, self._duration - 1)))

    def _clamp_frame(self, frame: int) -> int:
        if self._duration <= 0:
            return 0
        return max(0, min(frame, self._duration - 1))

    def _clamp_range(self, start: int, end: int) -> Tuple[int, int]:
        if self._duration <= 0:
            return 0, 0
        start = self._clamp_frame(start)
        end = self._clamp_frame(end)
        if end < start:
            end = start
        return start, end

    def _update_playhead_from_pos(self, pos: float, emit: bool = False) -> bool:
        frame = self._pos_to_frame(pos)
        return self._set_current_frame_value(frame, emit=emit)

    def _set_current_frame_value(self, frame: int, emit: bool = False) -> bool:
        frame = self._clamp_frame(frame)
        if frame == self._current:
            return False
        self._current = frame
        if emit:
            self.currentFrameChanged.emit(self._current)
        return True


__all__ = ["VideoFrameView", "ClipTimeline"]
