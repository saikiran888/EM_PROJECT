#!/usr/bin/env python3
"""
TEM / EM image folder viewer with SAM3 automatic segmentation, box-prompt
segmentation (drag a rectangle; each box adds contours), AnnotationMaster-style
contour interaction (click select, left-drag move, right-drag move, click empty
deselect, Ctrl multi-select), freehand / rubber sculpt / grow, copy/paste, class
keys 0–8, SAM mask assign, and JSON export.

Run from this project directory (script directory is added to sys.path automatically):

  cd /path/to/EMproject
  python tem_sam3_viewer.py

Environment:
  SAM3_FORCE_CPU=1       — force CPU
  SAM3_CHECKPOINT=/path/to/sam3.pt — override checkpoint (skips default search)
  SAM3_PATH=/path/to/sam3/repo     — SAM3 Python package root
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

# Project root (directory containing this script; holds Models/, Utils/)
_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

import copy

import numpy as np
import cv2
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QFileDialog,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMessageBox,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QShortcut,
    QComboBox,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, QTimer, QPointF, QLineF, QRectF

from Models.tem_edit_logic import TemContourEditLogic
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QKeySequence,
    QColor,
    QPen,
    QPainterPath,
    QCursor,
    QBrush,
    QFont,
)


def _fit_side_button(btn: QPushButton, *, max_width: int | None = None) -> None:
    """Size buttons from label text so captions are not clipped (Qt may shrink Maximum policy too far)."""
    fm = btn.fontMetrics()
    text = btn.text()
    try:
        tw = int(fm.horizontalAdvance(text))
    except Exception:
        tw = int(fm.boundingRect(text).width())
    pad = 22
    btn.setMinimumWidth(max(48, tw + pad))
    btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    btn.setMaximumHeight(30)
    if max_width is not None and max_width >= btn.minimumWidth():
        btn.setMaximumWidth(max_width)


def _color_for_class(class_name: str) -> QColor:
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    h = int(digest[:8], 16) % 360
    c = QColor()
    c.setHsv(h, 200, 255)
    return c


class ZoomGraphicsView(QGraphicsView):
    """Wheel zoom; pan (left on empty, middle always); SAM box/assign; P/R/O tools; navigate = AnnotationMaster-style contour edit."""

    def __init__(self, scene, on_zoom_callback=None):
        super().__init__(scene)
        self._on_zoom_callback = on_zoom_callback
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._panning_middle = False
        self._last_pan_pos = None
        # Box prompt: left drag = rectangle; middle = pan
        self._box_mode = False
        self._on_box_complete = None
        self._box_press_scene: QPointF | None = None
        self._box_rubber: QGraphicsRectItem | None = None
        # Click-to-assign class on SAM contours (left click); middle = pan
        self._assign_mode = False
        self._on_assign_click_cb = None
        # Freehand / rubber sculpt / grow-sh (AnnotationMaster-style); middle = pan
        self._contour_edit_active = False
        self._on_ce_press = None
        self._on_ce_move = None
        self._on_ce_release = None
        # Navigate mode: TemContourEditLogic select / drag / deselect (AnnotationMaster-style)
        self._on_nav_press = None
        self._on_nav_move = None
        self._on_nav_release = None
        self._on_nav_hover = None
        self.setMouseTracking(True)

    def set_navigate_contour_handlers(self, press=None, move=None, release=None):
        """When not in SAM box/assign or P/R/O tools, route mouse to edit logic (select, drag, deselect)."""
        self._on_nav_press = press
        self._on_nav_move = move
        self._on_nav_release = release

    def set_navigate_contour_hover(self, hover=None):
        """No-button mouse move: hover-select contour under cursor (requires mouse tracking)."""
        self._on_nav_hover = hover

    def set_contour_edit_active(self, active: bool):
        self._contour_edit_active = bool(active)
        if self._contour_edit_active:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CrossCursor))
        elif not self._box_mode and not self._assign_mode:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()
            self._panning_middle = False

    def set_contour_edit_mouse_handlers(self, press=None, move=None, release=None):
        self._on_ce_press = press
        self._on_ce_move = move
        self._on_ce_release = release

    def set_assign_mode(self, enabled: bool, on_click=None):
        """Single left-click in scene coordinates assigns class to a SAM contour under cursor."""
        self._assign_mode = bool(enabled)
        self._on_assign_click_cb = on_click
        if self._assign_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(QCursor(Qt.PointingHandCursor))
        elif not self._box_mode and not self._contour_edit_active:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()
            self._panning_middle = False

    def set_box_mode(self, enabled: bool, on_complete=None):
        """Drag left button to define a box in scene coordinates; on_complete(QRectF)."""
        self._box_mode = bool(enabled)
        self._on_box_complete = on_complete
        self._box_press_scene = None
        if self._box_rubber is not None:
            self.scene().removeItem(self._box_rubber)
            self._box_rubber = None
        if self._box_mode:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CrossCursor))
        elif not self._assign_mode and not self._contour_edit_active:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().unsetCursor()
            self._panning_middle = False

    def wheelEvent(self, event):
        if self._on_zoom_callback:
            self._on_zoom_callback()
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning_middle = True
            self._last_pan_pos = event.pos()
            event.accept()
            return
        if self._assign_mode:
            if event.button() == Qt.LeftButton:
                if self._on_assign_click_cb:
                    self._on_assign_click_cb(self.mapToScene(event.pos()))
                event.accept()
                return
        if self._box_mode:
            if event.button() == Qt.LeftButton:
                self._box_press_scene = self.mapToScene(event.pos())
                if self._box_rubber is not None:
                    self.scene().removeItem(self._box_rubber)
                self._box_rubber = QGraphicsRectItem(QRectF(self._box_press_scene, self._box_press_scene))
                pen = QPen(QColor(0, 200, 255), 2)
                pen.setCosmetic(True)
                self._box_rubber.setPen(pen)
                self._box_rubber.setBrush(QBrush(QColor(0, 200, 255, 25)))
                self._box_rubber.setZValue(6.0)
                self.scene().addItem(self._box_rubber)
                event.accept()
                return
        if self._contour_edit_active:
            if self._on_ce_press:
                self._on_ce_press(event)
            event.accept()
            return
        # Navigate: same pipeline as AnnotationMaster (edit_panel on_mouse_press / move / release).
        if self._on_nav_press and event.button() in (Qt.LeftButton, Qt.RightButton):
            if self._on_nav_press(event):
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._box_mode and self._box_press_scene is not None and self._box_rubber is not None:
            cur = self.mapToScene(event.pos())
            r = QRectF(self._box_press_scene, cur).normalized()
            self._box_rubber.setRect(r)
            event.accept()
            return
        if self._panning_middle and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        if (
            self._on_nav_hover
            and event.buttons() == Qt.NoButton
            and not self._assign_mode
            and not self._box_mode
            and not self._contour_edit_active
        ):
            self._on_nav_hover(event)
        if (
            self._on_nav_move
            and not self._assign_mode
            and not self._box_mode
            and not self._contour_edit_active
            and self._on_nav_move(event)
        ):
            event.accept()
            return
        if self._contour_edit_active and self._on_ce_move:
            self._on_ce_move(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._box_mode and event.button() == Qt.LeftButton and self._box_press_scene is not None:
            cur = self.mapToScene(event.pos())
            r = QRectF(self._box_press_scene, cur).normalized()
            self._box_press_scene = None
            if self._box_rubber is not None:
                self.scene().removeItem(self._box_rubber)
                self._box_rubber = None
            cb = self._on_box_complete
            if cb is not None and r.width() >= 4 and r.height() >= 4:
                cb(r)
            event.accept()
            return
        if self._contour_edit_active:
            if event.button() == Qt.MiddleButton:
                self._panning_middle = False
                self._last_pan_pos = None
                event.accept()
                return
            if self._on_ce_release and event.button() in (Qt.LeftButton, Qt.RightButton):
                self._on_ce_release(event)
                vp = self.viewport()
                if QWidget.mouseGrabber() is vp:
                    vp.releaseMouse()
                event.accept()
                return
        if (
            self._on_nav_release
            and not self._assign_mode
            and not self._box_mode
            and not self._contour_edit_active
            and event.button() in (Qt.LeftButton, Qt.RightButton)
        ):
            self._on_nav_release(event)
            vp = self.viewport()
            if QWidget.mouseGrabber() is vp:
                vp.releaseMouse()
            event.accept()
        if event.button() == Qt.MiddleButton:
            self._panning_middle = False
            self._last_pan_pos = None
        super().mouseReleaseEvent(event)

DEFAULT_IMAGE_DIR = (
    "/home/sai/Downloads/Niko TEM images/26-5 Niko/26-5_Niko_NEB_29T3"
)

# Class ids 0–8 for TEM organelles / structures (stored by name in JSON).
EM_CLASS_NAMES: list[str] = [
    "mitochondria_normal",
    "mitochondria_damaged",
    "nucleus",
    "condensed_chromatin",
    "autophagosome",
    "autolysosome",
    "apoptotic_body",
    "cell_membrane",
    "membrane_rupture",
]
# When SAM3_CHECKPOINT is unset, these paths are tried in order.
_CHECKPOINT_SEARCH_PATHS = [
    os.path.join(_APP_ROOT, "sam3.pt"),
    os.path.expanduser("~/Desktop/CellEventAnnotator/sam3.pt"),
]


def _resolve_sam3_checkpoint_path() -> str:
    """First existing path in _CHECKPOINT_SEARCH_PATHS, else the primary default (for errors)."""
    for p in _CHECKPOINT_SEARCH_PATHS:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return _CHECKPOINT_SEARCH_PATHS[0]


def load_tiff_u16(path: Path) -> np.ndarray:
    """Load single-page 16-bit (or 8-bit) grayscale TIFF as 2D uint16 or uint8."""
    path = Path(path)
    try:
        import tifffile

        data = tifffile.imread(str(path))
    except ImportError:
        from PIL import Image

        im = Image.open(str(path))
        data = np.array(im)
    if data.ndim == 3:
        if data.shape[2] >= 3:
            data = (0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2])
            data = data.astype(np.uint16 if data.dtype != np.uint8 else np.uint8)
        else:
            data = data[..., 0]
    if data.dtype == np.uint8:
        return data.astype(np.uint16)
    return data.astype(np.uint16)


def u16_to_rgb_u8(
    gray_u16: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> np.ndarray:
    """Percentile stretch to RGB uint8 for display and SAM."""
    g = gray_u16.astype(np.float32)
    lo = float(np.percentile(g, p_low))
    hi = float(np.percentile(g, p_high))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    u8 = (scaled * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def limit_long_edge(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    """Resize so max(h,w) <= max_side. Returns (rgb, scale_x, scale_y) to map model coords to full image."""
    h, w = rgb.shape[:2]
    if max_side <= 0:
        return rgb, 1.0, 1.0
    m = max(h, w)
    if m <= max_side:
        return rgb, 1.0, 1.0
    scale = max_side / m
    nw, nh = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    sx = w / nw
    sy = h / nh
    return small, sx, sy


def normalize_masks(masks) -> list:
    """Accept list of dicts (SAM3) or ndarrays; return list of bool (H,W) arrays."""
    out = []
    if masks is None:
        return out
    if isinstance(masks, np.ndarray):
        masks = [masks]
    for m in masks:
        if isinstance(m, dict):
            seg = m.get("segmentation", None)
            if seg is None:
                continue
        else:
            seg = m
        if torch_is_tensor(seg):
            seg = seg.detach().cpu().numpy()
        seg = np.asarray(seg).squeeze()
        if seg.ndim != 2:
            continue
        out.append(seg.astype(bool))
    return out


def torch_is_tensor(x):
    try:
        import torch

        return torch.is_tensor(x)
    except ImportError:
        return False


def scale_mask_to_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize boolean mask to (height, width)."""
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask
    u8 = (mask.astype(np.uint8) * 255)
    resized = cv2.resize(u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def draw_contours_rgb(
    rgb: np.ndarray, masks: list, bgr_color=(0, 255, 0), thickness: int = 2
) -> np.ndarray:
    """Draw external contours on RGB image (OpenCV uses BGR for color tuple)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]
    for mask in masks:
        m = scale_mask_to_shape(mask, w, h)
        u8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, bgr_color, thickness)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for boolean masks of the same shape."""
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    if inter <= 0:
        return 0.0
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / max(union, 1.0))


def mask_to_closed_contour_xy(mask: np.ndarray) -> list[list[float]]:
    """Largest external contour of a boolean mask → closed [[x,y],…] for JSON."""
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return []
    peri = float(cv2.arcLength(c, True))
    eps = max(1.0, 0.0015 * peri)
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2)
    out = [[round(float(x), 2), round(float(y), 2)] for x, y in pts]
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    h, w, ch = rgb.shape
    if ch != 3:
        raise ValueError("RGB array expected")
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class TemSam3ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEM SAM3 viewer")
        self.resize(1400, 900)

        self._folder = Path(DEFAULT_IMAGE_DIR)
        self._files: list[Path] = []
        self._index = -1
        self._raw_u16: np.ndarray | None = None
        self._display_rgb: np.ndarray | None = None
        self._last_model_rgb: np.ndarray | None = None
        self._last_scale: tuple[float, float] = (1.0, 1.0)
        self._overlay_rgb: np.ndarray | None = None
        self._user_zoomed = False
        # SAM instances waiting for class (display-sized bool masks); box prompts append here.
        self._sam_unlabeled_masks: list[np.ndarray] = []

        # Manual labels: folder JSON + per-image list of {class, contour: [[x,y],...]}
        self._annotations_path: Path | None = None
        self._annotations: dict = {"version": 1, "images": {}}
        self._anno_group: QGraphicsItemGroup | None = None
        self._preview_lines: list[QGraphicsLineItem] = []
        self._preview_close_line: QGraphicsLineItem | None = None

        self._edit = TemContourEditLogic(self)
        self._copied_ann: dict | None = None

        self.sam_worker = None
        self.sam_thread: QThread | None = None
        self.sam_loaded = False
        self._infer_busy = False

        self._build_ui()
        self._populate_file_list()

        if self._files:
            self._load_index(0)

        self._wire_contour_editor()

        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._next)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._prev)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._fit_image_view)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._esc_shortcut)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_selected_annotation)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self._delete_selected_annotation)

    def _esc_shortcut(self):
        self._clear_poly_preview()
        self._edit.clear_editing_state()
        self.status.setText("Cancelled in-progress edit.")

    def _wire_contour_editor(self):
        self._edit.contoursUpdated.connect(self._on_edit_contours_updated)
        self._edit.selectionChanged.connect(self._on_edit_selection_changed)
        self._edit.freehandPreviewUpdated.connect(self._on_edit_freehand_preview)
        self._edit.set_data_accessors(
            self._edit_get_contours, self._edit_set_contour, self._edit_add_contour
        )
        self.view.set_contour_edit_mouse_handlers(
            self._handle_ce_press, self._handle_ce_move, self._handle_ce_release
        )
        self.view.set_navigate_contour_handlers(
            self._handle_nav_press, self._handle_nav_move, self._handle_nav_release
        )
        self.view.set_navigate_contour_hover(self._handle_nav_hover)

        QShortcut(QKeySequence("P"), self, activated=lambda: self._shortcut_toggle_mode("freehand"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self._shortcut_toggle_mode("rubber"))
        QShortcut(QKeySequence("O"), self, activated=lambda: self._shortcut_toggle_mode("grow"))
        QShortcut(QKeySequence.Copy, self, activated=self._copy_selected_contour)
        QShortcut(QKeySequence.Paste, self, activated=self._paste_copied_contour)
        for i in range(9):
            QShortcut(QKeySequence(str(i)), self, activated=lambda n=i: self._apply_class_digit(n))

        # Alt+arrows nudge in rubber mode (plain ←/→ still change image)
        QShortcut(QKeySequence("Alt+Left"), self, activated=lambda: self._edit_arrow_nudge(-1, 0))
        QShortcut(QKeySequence("Alt+Right"), self, activated=lambda: self._edit_arrow_nudge(1, 0))
        QShortcut(QKeySequence("Alt+Up"), self, activated=lambda: self._edit_arrow_nudge(0, -1))
        QShortcut(QKeySequence("Alt+Down"), self, activated=lambda: self._edit_arrow_nudge(0, 1))

    def _edit_ann_indices_valid(self) -> list[int]:
        return [
            i
            for i, ann in enumerate(self._annotations_for_current())
            if len(ann.get("contour", [])) >= 3
        ]

    def _edit_get_contours(self) -> list:
        lst = self._annotations_for_current()
        out = []
        for i in self._edit_ann_indices_valid():
            c = lst[i].get("contour", [])
            out.append(np.array(c, dtype=np.int32).reshape(-1, 1, 2))
        return out

    def _np_contour_to_xy_list(self, arr: np.ndarray) -> list[list[float]]:
        if arr.ndim == 3:
            pts = arr.reshape(-1, 2)
        else:
            pts = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
        return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]

    def _edit_set_contour(self, edit_idx: int, arr: np.ndarray) -> None:
        vi = self._edit_ann_indices_valid()
        if edit_idx < 0 or edit_idx >= len(vi):
            return
        self._annotations_for_current()[vi[edit_idx]]["contour"] = self._np_contour_to_xy_list(arr)

    def _edit_add_contour(self, arr: np.ndarray) -> None:
        cls = self._class_name_from_combo()
        self._annotations_for_current().append(
            {
                "class": cls,
                "contour": self._np_contour_to_xy_list(arr),
                "source": "freehand",
            }
        )

    def _on_edit_contours_updated(self):
        self._populate_annotation_list_widget()
        self._redraw_annotation_items()

    def _on_edit_selection_changed(self, idx: int, _n: int = 0):
        vi = self._edit_ann_indices_valid()
        if idx < 0 or idx >= len(vi):
            row = -1
        else:
            row = vi[idx]
        self.list_annotations.blockSignals(True)
        self.list_annotations.setCurrentRow(row)
        self.list_annotations.blockSignals(False)
        if row >= 0:
            ann = self._annotations_for_current()[row]
            cls = str(ann.get("class", ""))
            self.combo_class.blockSignals(True)
            if cls in EM_CLASS_NAMES:
                self.combo_class.setCurrentIndex(EM_CLASS_NAMES.index(cls))
            self.combo_class.blockSignals(False)
            dk = self._class_digit_for_name(cls)
            self.status.setText(
                f"Selected #{row + 1}: {cls}  [dropdown or {dk}]  "
                f"(hover or click contour, drag to move, Delete; P/R/O tools; Ctrl+click multi; Ctrl+C/V)"
            )
        else:
            self.status.setText("No contour selected (hover/click a contour or pick from the list).")
        self._redraw_annotation_items()

    def _class_digit_for_name(self, cls: str) -> str:
        try:
            return str(EM_CLASS_NAMES.index(cls))
        except ValueError:
            return "?"

    def _on_combo_class_changed(self, idx: int):
        if idx < 0 or idx >= len(EM_CLASS_NAMES):
            return
        ei = self._edit.get_selected_contour_index()
        if ei is None:
            return
        vi = self._edit_ann_indices_valid()
        if ei < 0 or ei >= len(vi):
            return
        row = vi[ei]
        cls = EM_CLASS_NAMES[idx]
        self._annotations_for_current()[row]["class"] = cls
        self._populate_annotation_list_widget()
        self._redraw_annotation_items()
        self.status.setText(f"Class set — {cls}  (#{row + 1})")

    def _on_edit_freehand_preview(self, pts: list):
        self._clear_poly_preview()
        if len(pts) < 2:
            return
        pen = QPen(QColor(255, 220, 0), 2)
        pen.setCosmetic(True)
        for i in range(len(pts) - 1):
            a = QPointF(float(pts[i][0]), float(pts[i][1]))
            b = QPointF(float(pts[i + 1][0]), float(pts[i + 1][1]))
            ln = QGraphicsLineItem(QLineF(a, b))
            ln.setPen(pen)
            ln.setZValue(5.0)
            self.scene.addItem(ln)
            self._preview_lines.append(ln)

    def _handle_nav_press(self, event) -> bool:
        """AnnotationMaster-style: select / drag / deselect via edit_logic. Returns True if Qt hand-drag must not run."""
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        xi, yi = int(p.x()), int(p.y())
        if event.button() == Qt.LeftButton:
            self._edit.on_mouse_press(
                1,
                xi,
                yi,
                shift_modifier=bool(event.modifiers() & Qt.ControlModifier),
            )
            if self._edit.is_drag_move_active() or self._edit.is_rubber_band_drag_active():
                self.view.viewport().grabMouse()
                return True
            if self._edit.get_target_contour_index(xi, yi) is not None:
                return True
            return False
        if event.button() == Qt.RightButton:
            self._edit.on_mouse_press(2, xi, yi)
            if self._edit.is_drag_move_active():
                self.view.viewport().grabMouse()
            return True
        return False

    def _handle_nav_move(self, event) -> bool:
        if not (self._edit.is_drag_move_active() or self._edit.is_rubber_band_drag_active()):
            return False
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        self._edit.on_mouse_move(int(p.x()), int(p.y()))
        return True

    def _handle_nav_hover(self, event) -> None:
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        self._edit.on_hover(int(p.x()), int(p.y()))

    def _handle_nav_release(self, event):
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        if event.button() == Qt.LeftButton:
            btn = 1
        elif event.button() == Qt.RightButton:
            btn = 2
        else:
            btn = 0
        self._edit.on_mouse_release(btn, int(p.x()), int(p.y()))

    def _handle_ce_press(self, event):
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        if event.button() == Qt.LeftButton:
            btn = 1
        elif event.button() == Qt.RightButton:
            btn = 2
        else:
            return
        shift = bool(event.modifiers() & Qt.ControlModifier)
        self._edit.on_mouse_press(btn, int(p.x()), int(p.y()), shift_modifier=shift)
        if self._edit.is_rubber_band_drag_active() or self._edit.is_drag_move_active():
            self.view.viewport().grabMouse()

    def _handle_ce_move(self, event):
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        self._edit.on_mouse_move(int(p.x()), int(p.y()))

    def _handle_ce_release(self, event):
        p = self._clamp_scene_point(self.view.mapToScene(event.pos()))
        if event.button() == Qt.LeftButton:
            btn = 1
        elif event.button() == Qt.RightButton:
            btn = 2
        else:
            btn = 0
        self._edit.on_mouse_release(btn, int(p.x()), int(p.y()))
        self._clear_poly_preview()

    def _sync_contour_edit_view_mode(self):
        active = (
            self.btn_freehand.isChecked()
            or self.btn_rubber.isChecked()
            or self.btn_grow.isChecked()
        )
        self.view.set_contour_edit_active(active)

    def _uncheck_edit_tool_buttons(self, except_name: str | None = None):
        for name, btn in (
            ("freehand", self.btn_freehand),
            ("rubber", self.btn_rubber),
            ("grow", self.btn_grow),
        ):
            if except_name == name:
                continue
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

    def _on_edit_mode_toggled(self, mode: str, checked: bool):
        if not checked:
            self._edit.set_modes(freehand=False, rubber_edit=False, region_growth=False)
            self._sync_contour_edit_view_mode()
            return
        self._uncheck_edit_tool_buttons(except_name=mode)
        self._edit.set_modes(
            freehand=(mode == "freehand"),
            rubber_edit=(mode == "rubber"),
            region_growth=(mode == "grow"),
        )
        if self.btn_toggle_assign.isChecked():
            self.btn_toggle_assign.blockSignals(True)
            self.btn_toggle_assign.setChecked(False)
            self.btn_toggle_assign.blockSignals(False)
        self.view.set_assign_mode(False)
        if self.btn_toggle_box.isChecked():
            self.btn_toggle_box.blockSignals(True)
            self.btn_toggle_box.setChecked(False)
            self.btn_toggle_box.blockSignals(False)
        self.view.set_box_mode(False)
        self._sync_contour_edit_view_mode()
        tips = {
            "freehand": "Freehand (P): left-drag draw, release saves with class above. Right = select+drag. Middle = pan.",
            "rubber": "Rubber sculpt (R): click contour to select, drag near outline. Alt+arrows nudge. Middle = pan.",
            "grow": "Grow / shrink (O): click contour to select, then left-click near edge. Middle = pan.",
        }
        self.status.setText(tips.get(mode, ""))

    def _shortcut_toggle_mode(self, mode: str):
        btn = {"freehand": self.btn_freehand, "rubber": self.btn_rubber, "grow": self.btn_grow}[mode]
        btn.setChecked(not btn.isChecked())

    def _edit_arrow_nudge(self, dx: int, dy: int):
        if not self._edit.is_rubber_edit() or self._edit.get_selected_contour_index() is None:
            return
        step = self._edit.get_nudge_step()
        self._edit.apply_nudge(dx * step, dy * step)

    def _apply_class_digit(self, digit: int):
        if digit < 0 or digit >= len(EM_CLASS_NAMES):
            return
        ei = self._edit.get_selected_contour_index()
        if ei is None:
            self.status.setText("Select a contour first (click contour or list), then press 0–8.")
            return
        vi = self._edit_ann_indices_valid()
        if ei < 0 or ei >= len(vi):
            return
        row = vi[ei]
        cls = EM_CLASS_NAMES[digit]
        self._annotations_for_current()[row]["class"] = cls
        self.combo_class.blockSignals(True)
        self.combo_class.setCurrentIndex(digit)
        self.combo_class.blockSignals(False)
        self._populate_annotation_list_widget()
        self._redraw_annotation_items()
        self.status.setText(f"Class set to {digit} — {cls}")

    def _copy_selected_contour(self):
        ei = self._edit.get_selected_contour_index()
        if ei is None:
            self.status.setText("Select a contour to copy (click contour or list).")
            return
        vi = self._edit_ann_indices_valid()
        if ei < 0 or ei >= len(vi):
            return
        self._copied_ann = copy.deepcopy(self._annotations_for_current()[vi[ei]])
        self.status.setText("Copied contour (Ctrl+V to paste).")

    def _paste_copied_contour(self):
        if not self._copied_ann:
            self.status.setText("Nothing copied.")
            return
        self._annotations_for_current().append(copy.deepcopy(self._copied_ann))
        self._populate_annotation_list_widget()
        self._edit.clear_selection(silent=True)
        self._redraw_annotation_items()
        self.status.setText("Pasted contour.")

    def _on_annotation_list_row_changed(self, row: int):
        if row < 0:
            self._edit.set_selected_contour_index(None, silent=True)
            self._redraw_annotation_items()
            return
        vi = self._edit_ann_indices_valid()
        if row in vi:
            self._edit.set_selected_contour_index(vi.index(row), silent=True)
        self._redraw_annotation_items()

    def _stop_all_contour_tools(self):
        for btn in (self.btn_freehand, self.btn_rubber, self.btn_grow):
            if btn.isChecked():
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        self._edit.clear_editing_state()
        self._edit.clear_selection(silent=True)
        self._clear_poly_preview()
        self.view.set_contour_edit_active(False)

    def _on_view_zoomed(self):
        self._user_zoomed = True

    def _nudge_zoom(self, factor: float):
        self._on_view_zoomed()
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.view.scale(factor, factor)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)

        # Left: list + controls (narrow column; buttons sized to content)
        left = QWidget()
        left.setMaximumWidth(380)
        left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_l = QVBoxLayout(left)
        left_l.setSpacing(6)

        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        left_l.addWidget(self.folder_label)

        btn_folder = QPushButton("Open folder…")
        btn_folder.clicked.connect(self._pick_folder)
        left_l.addWidget(btn_folder)

        self.list_w = QListWidget()
        self.list_w.currentRowChanged.connect(self._on_row_changed)
        left_l.addWidget(self.list_w, 1)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)
        _fit_side_button(self.btn_prev)
        _fit_side_button(self.btn_next)
        nav.addWidget(self.btn_prev, 0, Qt.AlignLeft)
        nav.addWidget(self.btn_next, 0, Qt.AlignLeft)
        nav.addStretch(1)
        left_l.addLayout(nav)

        grp_img = QGroupBox("Preprocess")
        fl = QFormLayout(grp_img)
        self.sp_p_low = QDoubleSpinBox()
        self.sp_p_low.setRange(0.0, 50.0)
        self.sp_p_low.setValue(1.0)
        self.sp_p_low.setSingleStep(0.5)
        fl.addRow("Stretch low %", self.sp_p_low)

        self.sp_p_high = QDoubleSpinBox()
        self.sp_p_high.setRange(50.0, 100.0)
        self.sp_p_high.setValue(99.0)
        self.sp_p_high.setSingleStep(0.5)
        fl.addRow("Stretch high %", self.sp_p_high)
        self.sp_p_low.valueChanged.connect(self._on_preprocess_changed)
        self.sp_p_high.valueChanged.connect(self._on_preprocess_changed)

        self.sp_max_side = QSpinBox()
        self.sp_max_side.setRange(0, 8192)
        self.sp_max_side.setValue(2048)
        self.sp_max_side.setSpecialValueText("off")
        self.sp_max_side.setToolTip("0 = full resolution to SAM (slow). Otherwise long edge limit.")
        fl.addRow("Max side (px)", self.sp_max_side)

        left_l.addWidget(grp_img)

        grp_sam = QGroupBox("SAM3")
        fl2 = QFormLayout(grp_sam)
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.05, 0.95)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.35)
        self.sp_conf.setToolTip("Passed to worker; SAM3 automatic path may cap lower internally.")
        fl2.addRow("Confidence", self.sp_conf)

        self.sp_max_cells = QSpinBox()
        self.sp_max_cells.setRange(1, 200)
        self.sp_max_cells.setValue(48)
        fl2.addRow("Max instances", self.sp_max_cells)

        self.combo_mask_mode = QComboBox()
        self.combo_mask_mode.addItem("Mixed — large + small", "mixed")
        self.combo_mask_mode.addItem("Largest regions first", "largest")
        self.combo_mask_mode.addItem("Highest score first", "score")
        self.combo_mask_mode.addItem("Smallest only (original)", "smallest")
        self.combo_mask_mode.setToolTip(
            "How to choose instances after SAM3 filtering. "
            "Use Mixed or Largest to see big cell outlines, not only vesicles."
        )
        fl2.addRow("Mask pick", self.combo_mask_mode)
        self.combo_mask_mode.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.btn_load_sam = QPushButton("Load SAM3")
        self.btn_load_sam.setToolTip("Load SAM3 model checkpoint")
        self.btn_load_sam.clicked.connect(self._on_load_sam)
        self.btn_run = QPushButton("Run SAM")
        self.btn_run.setToolTip("Run full-image segmentation")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save = QPushButton("Save PNG…")
        self.btn_save.setToolTip("SAM contours (if any) plus your labeled freehand contours")
        self.btn_save.clicked.connect(self._on_save_overlay)
        for b in (self.btn_load_sam, self.btn_run, self.btn_save):
            _fit_side_button(b)
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(0)
            row_l.addWidget(b, 0, Qt.AlignLeft)
            row_l.addStretch(1)
            fl2.addRow(row_w)

        left_l.addWidget(grp_sam)

        grp_ann = QGroupBox("Classes & contours")
        ann_l = QVBoxLayout(grp_ann)
        self.combo_class = QComboBox()
        for i, name in enumerate(EM_CLASS_NAMES):
            self.combo_class.addItem(f"{i} — {name}", name)
        self.combo_class.setCurrentIndex(0)
        ann_l.addWidget(
            QLabel(
                "Class 0–8: click a contour to select (or list); change dropdown or press 0–8 to set class. "
                "SAM green masks: turn on “SAM class”, then click the contour."
            )
        )
        ann_l.addWidget(self.combo_class)
        self.combo_class.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.combo_class.currentIndexChanged.connect(self._on_combo_class_changed)

        row_draw1 = QHBoxLayout()
        row_draw1.setSpacing(6)
        row_draw1b = QHBoxLayout()
        row_draw1b.setSpacing(6)
        row_draw2 = QHBoxLayout()
        row_draw2.setSpacing(6)
        self.btn_freehand = QPushButton("Draw (P)")
        self.btn_freehand.setCheckable(True)
        self.btn_freehand.setToolTip(
            "Freehand (P): draw a new contour; release saves with class above. Middle = pan."
        )
        self.btn_freehand.toggled.connect(lambda c: self._on_edit_mode_toggled("freehand", c))
        self.btn_rubber = QPushButton("Sculpt (R)")
        self.btn_rubber.setCheckable(True)
        self.btn_rubber.setToolTip(
            "Rubber sculpt (R): click contour to select, drag near outline; Alt+arrows nudge. Middle = pan."
        )
        self.btn_rubber.toggled.connect(lambda c: self._on_edit_mode_toggled("rubber", c))
        self.btn_grow = QPushButton("Grow (O)")
        self.btn_grow.setCheckable(True)
        self.btn_grow.setToolTip(
            "Grow/shrink (O): click contour to select, then left-click near edge. Middle = pan."
        )
        self.btn_grow.toggled.connect(lambda c: self._on_edit_mode_toggled("grow", c))
        self.btn_copy_ann = QPushButton("Copy")
        self.btn_copy_ann.setToolTip("Copy selected contour (Ctrl+C). Select by clicking the contour or list.")
        self.btn_copy_ann.clicked.connect(self._copy_selected_contour)
        self.btn_paste_ann = QPushButton("Paste")
        self.btn_paste_ann.setToolTip("Paste copied contour (Ctrl+V)")
        self.btn_paste_ann.clicked.connect(self._paste_copied_contour)
        self.btn_toggle_box = QPushButton("Box")
        self.btn_toggle_box.setCheckable(True)
        self.btn_toggle_box.setEnabled(False)
        self.btn_toggle_box.setToolTip(
            "Box prompt: drag a rectangle; SAM3 segments inside. Middle-drag to pan."
        )
        self.btn_toggle_box.toggled.connect(self._on_toggle_box)
        self.btn_toggle_assign = QPushButton("SAM class")
        self.btn_toggle_assign.setCheckable(True)
        self.btn_toggle_assign.setEnabled(False)
        self.btn_toggle_assign.setToolTip(
            "Assign class by clicking a green SAM contour (or near center). Middle-drag pans."
        )
        self.btn_toggle_assign.toggled.connect(self._on_toggle_assign)
        self.btn_cancel_poly = QPushButton("Cancel")
        self.btn_cancel_poly.clicked.connect(self._esc_shortcut)
        self.btn_cancel_poly.setToolTip("Esc: cancel in-progress freehand / sculpt drag.")
        for b in (
            self.btn_freehand,
            self.btn_rubber,
            self.btn_grow,
            self.btn_copy_ann,
            self.btn_paste_ann,
            self.btn_toggle_box,
            self.btn_toggle_assign,
            self.btn_cancel_poly,
        ):
            _fit_side_button(b)
        for b in (self.btn_freehand, self.btn_rubber, self.btn_grow):
            row_draw1.addWidget(b, 0, Qt.AlignLeft)
        row_draw1.addStretch(1)
        for b in (self.btn_copy_ann, self.btn_paste_ann):
            row_draw1b.addWidget(b, 0, Qt.AlignLeft)
        row_draw1b.addStretch(1)
        for b in (self.btn_toggle_box, self.btn_toggle_assign, self.btn_cancel_poly):
            row_draw2.addWidget(b, 0, Qt.AlignLeft)
        row_draw2.addStretch(1)
        ann_l.addLayout(row_draw1)
        ann_l.addLayout(row_draw1b)
        ann_l.addLayout(row_draw2)

        ann_l.addWidget(QLabel("Annotations (this image):"))
        self.list_annotations = QListWidget()
        self.list_annotations.setMinimumHeight(100)
        self.list_annotations.currentRowChanged.connect(self._on_annotation_list_row_changed)
        ann_l.addWidget(self.list_annotations)

        row_del = QHBoxLayout()
        row_del.setSpacing(4)
        self.btn_del_ann = QPushButton("Delete")
        self.btn_del_ann.setToolTip("Delete selected contour (click contour first, or list). Delete/Backspace.")
        self.btn_del_ann.clicked.connect(self._delete_selected_annotation)
        _fit_side_button(self.btn_del_ann)
        row_del.addWidget(self.btn_del_ann, 0, Qt.AlignLeft)
        row_del.addStretch(1)
        ann_l.addLayout(row_del)

        row_io = QHBoxLayout()
        row_io.setSpacing(4)
        self.btn_save_json = QPushButton("Save JSON")
        self.btn_save_json.setToolTip("Save labels JSON to folder")
        self.btn_save_json.clicked.connect(self._save_annotations_json)
        self.btn_load_json = QPushButton("Reload")
        self.btn_load_json.setToolTip("Reload annotations_tem.json")
        self.btn_load_json.clicked.connect(self._reload_annotations_json)
        for b in (self.btn_save_json, self.btn_load_json):
            _fit_side_button(b)
        row_io.addWidget(self.btn_save_json, 0, Qt.AlignLeft)
        row_io.addWidget(self.btn_load_json, 0, Qt.AlignLeft)
        row_io.addStretch(1)
        ann_l.addLayout(row_io)

        hint_ann = QLabel(
            "P freehand · R sculpt · O grow · click contour (Ctrl=multi) · drag to move · right-drag move · "
            "0–8 class · Delete · Ctrl+C/V. Empty+left = pan; middle-drag = pan anytime. Box/Assign = SAM."
        )
        hint_ann.setWordWrap(True)
        hint_ann.setStyleSheet("color: #666; font-size: 11px;")
        ann_l.addWidget(hint_ann)

        left_l.addWidget(grp_ann)

        self.status = QLabel("Open a folder and load SAM3.")
        self.status.setWordWrap(True)
        left_l.addWidget(self.status)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit view — show whole image")
        self.btn_fit.clicked.connect(self._fit_image_view)
        self.btn_zin = QPushButton("+")
        self.btn_zin.setToolTip("Zoom in")
        self.btn_zin.clicked.connect(lambda: self._nudge_zoom(1.2))
        self.btn_zout = QPushButton("−")
        self.btn_zout.setToolTip("Zoom out")
        self.btn_zout.clicked.connect(lambda: self._nudge_zoom(1.0 / 1.2))
        for b in (self.btn_fit, self.btn_zout, self.btn_zin):
            _fit_side_button(b)
        zoom_row.addWidget(self.btn_fit, 0, Qt.AlignLeft)
        zoom_row.addWidget(self.btn_zout, 0, Qt.AlignLeft)
        zoom_row.addWidget(self.btn_zin, 0, Qt.AlignLeft)
        zoom_row.addStretch(1)
        left_l.addLayout(zoom_row)
        zhint = QLabel(
            "Tip: wheel zoom; left-drag pans when no edit tool is on. With P/R/O or box/assign, middle-drag pans."
        )
        zhint.setWordWrap(True)
        zhint.setStyleSheet("color: #666; font-size: 11px;")
        left_l.addWidget(zhint)

        splitter.addWidget(left)

        # Right: graphics view (wheel zoom + freehand / box draw modes)
        self.scene = QGraphicsScene(self)
        self.view = ZoomGraphicsView(self.scene, on_zoom_callback=self._on_view_zoomed)
        self.pix_item = QGraphicsPixmapItem()
        # Let left-clicks reach the view for pan/zoom; contours sit above in _anno_group
        self.pix_item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene.addItem(self.pix_item)
        self._anno_group = QGraphicsItemGroup()
        self._anno_group.setZValue(2.0)
        self.pix_item.setZValue(0.0)
        self.scene.addItem(self._anno_group)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1020])

        main.addWidget(splitter)
        self._refresh_folder_label()

    def _on_preprocess_changed(self):
        if self._raw_u16 is None:
            return
        self._sam_unlabeled_masks.clear()
        self._overlay_rgb = None
        self._refresh_display()

    def _refresh_folder_label(self):
        self.folder_label.setText(f"Folder:\n{self._folder}")

    def _current_image_key(self) -> str | None:
        if self._index < 0 or self._index >= len(self._files):
            return None
        return self._files[self._index].name

    def _try_load_annotations_file(self):
        self._annotations_path = self._folder / "annotations_tem.json"
        if self._annotations_path.is_file():
            try:
                with open(self._annotations_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("images"), dict):
                    self._annotations = data
                    self._annotations.setdefault("version", 1)
                else:
                    self._annotations = {"version": 1, "images": {}}
            except (json.JSONDecodeError, OSError):
                self._annotations = {"version": 1, "images": {}}
        else:
            self._annotations = {"version": 1, "images": {}}

    def _annotations_for_current(self) -> list:
        key = self._current_image_key()
        if not key:
            return []
        imgs = self._annotations.setdefault("images", {})
        if key not in imgs:
            imgs[key] = []
        if not isinstance(imgs[key], list):
            imgs[key] = []
        return imgs[key]

    def _populate_annotation_list_widget(self):
        self.list_annotations.clear()
        for i, ann in enumerate(self._annotations_for_current()):
            cls = ann.get("class", "?")
            npt = len(ann.get("contour", []))
            self.list_annotations.addItem(f"{i + 1}. [{cls}]  {npt} points")

    def _class_name_from_combo(self) -> str:
        data = self.combo_class.currentData()
        if data is not None and str(data).strip():
            return str(data).strip()
        t = self.combo_class.currentText().strip()
        return t or EM_CLASS_NAMES[0]

    def _worker_masks_to_display(self, masks_norm: list) -> list[np.ndarray]:
        if self._display_rgb is None:
            return []
        h, w = self._display_rgb.shape[:2]
        scaled: list[np.ndarray] = []
        for m in masks_norm:
            mh, mw = m.shape[:2]
            if self._last_model_rgb is not None and (mh, mw) == (
                self._last_model_rgb.shape[0],
                self._last_model_rgb.shape[1],
            ):
                u8 = (m.astype(np.uint8) * 255)
                resized = cv2.resize(u8, (w, h), interpolation=cv2.INTER_NEAREST)
                scaled.append(resized.astype(bool))
            else:
                scaled.append(scale_mask_to_shape(m, w, h))
        return scaled

    def _dedupe_append_masks(self, new_masks: list[np.ndarray]) -> tuple[int, int]:
        """Append masks not heavily overlapping existing unlabeled masks. Returns (added, skipped)."""
        added = 0
        skipped = 0
        for m in new_masks:
            if any(mask_iou(m, ex) > 0.88 for ex in self._sam_unlabeled_masks):
                skipped += 1
                continue
            self._sam_unlabeled_masks.append(m.copy())
            added += 1
        return added, skipped

    def _rebuild_sam_overlay(self, status_msg: str) -> None:
        if self._display_rgb is None:
            return
        if not self._sam_unlabeled_masks:
            self._overlay_rgb = None
            self.pix_item.setPixmap(numpy_rgb_to_qpixmap(self._display_rgb))
        else:
            self._overlay_rgb = draw_contours_rgb(
                self._display_rgb,
                self._sam_unlabeled_masks,
                bgr_color=(0, 255, 0),
                thickness=2,
            )
            self.pix_item.setPixmap(numpy_rgb_to_qpixmap(self._overlay_rgb))
        self._redraw_annotation_items()
        self.status.setText(status_msg)

    def _pick_sam_mask_index(self, x: float, y: float) -> int:
        if self._display_rgb is None or not self._sam_unlabeled_masks:
            return -1
        h, w = self._display_rgb.shape[:2]
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            for i in range(len(self._sam_unlabeled_masks) - 1, -1, -1):
                if self._sam_unlabeled_masks[i][yi, xi]:
                    return i
        best_i = -1
        best_d = 28.0**2
        for i, m in enumerate(self._sam_unlabeled_masks):
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _on_scene_assign_class_click(self, pos: QPointF) -> None:
        if self._display_rgb is None:
            return
        if not self._sam_unlabeled_masks:
            self.status.setText("No green SAM contours. Run segmentation or add a box prompt first.")
            return
        p = self._clamp_scene_point(pos)
        idx = self._pick_sam_mask_index(p.x(), p.y())
        if idx < 0:
            self.status.setText("No contour under click — try the green outline or its center.")
            return
        m = self._sam_unlabeled_masks.pop(idx)
        contour = mask_to_closed_contour_xy(m)
        if len(contour) < 3:
            self._sam_unlabeled_masks.insert(idx, m)
            self.status.setText("Could not extract a contour from that mask.")
            return
        cls = self._class_name_from_combo()
        self._annotations_for_current().append(
            {"class": cls, "contour": contour, "source": "sam_box"}
        )
        self._populate_annotation_list_widget()
        self._rebuild_sam_overlay(
            f"Saved “{cls}”. {len(self._sam_unlabeled_masks)} green SAM contour(s) left."
        )

    def _on_toggle_assign(self, checked: bool):
        if checked:
            self._stop_all_contour_tools()
            if self.btn_toggle_box.isChecked():
                self.btn_toggle_box.blockSignals(True)
                self.btn_toggle_box.setChecked(False)
                self.btn_toggle_box.blockSignals(False)
            self.view.set_box_mode(False)
            self.view.set_assign_mode(True, on_click=self._on_scene_assign_class_click)
            self.status.setText(
                "SAM class mode: click a green contour (or near center). Class = dropdown. Middle-drag pans."
            )
        else:
            self.view.set_assign_mode(False)
            self.status.setText("SAM class (click) mode off.")

    def _rebuild_anno_group(self):
        if self._anno_group is not None:
            self.scene.removeItem(self._anno_group)
        self._anno_group = QGraphicsItemGroup()
        self._anno_group.setZValue(2.0)
        self.scene.addItem(self._anno_group)

    def _redraw_annotation_items(self):
        self._rebuild_anno_group()
        if self._display_rgb is None:
            return
        key = self._current_image_key()
        if not key:
            return
        sel_row = None
        ei = self._edit.get_selected_contour_index()
        if ei is not None:
            vi = self._edit_ann_indices_valid()
            if 0 <= ei < len(vi):
                sel_row = vi[ei]
        lst = self._annotations.get("images", {}).get(key, [])
        for row_idx, ann in enumerate(lst):
            pts = ann.get("contour", [])
            if len(pts) < 3:
                continue
            path = QPainterPath()
            path.moveTo(pts[0][0], pts[0][1])
            for j in range(1, len(pts)):
                path.lineTo(pts[j][0], pts[j][1])
            path.closeSubpath()
            item = QGraphicsPathItem(path)
            col = _color_for_class(str(ann.get("class", "")))
            sel = row_idx == sel_row
            # Selected: high-contrast gold outline; unselected: class color
            sel_pen = QColor(255, 200, 40)
            pen = QPen(sel_pen if sel else col, 5 if sel else 2)
            pen.setCosmetic(True)
            if sel:
                pen.setJoinStyle(Qt.RoundJoin)
                pen.setCapStyle(Qt.RoundCap)
            item.setPen(pen)
            item.setBrush(
                QBrush(QColor(col.red(), col.green(), col.blue(), 70 if sel else 40))
            )
            self._anno_group.addToGroup(item)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cls = str(ann.get("class", ""))
            dk = self._class_digit_for_name(cls)
            lbl = f"{cls}  [{dk}]"
            ti = QGraphicsTextItem(lbl)
            f = QFont()
            f.setPointSize(9)
            ti.setFont(f)
            ti.setDefaultTextColor(col.darker(140))
            br = ti.boundingRect()
            ti.setPos(QPointF(cx - br.width() / 2.0, cy - br.height() / 2.0))
            ti.setZValue(3.0)
            ti.setAcceptedMouseButtons(Qt.NoButton)
            self._anno_group.addToGroup(ti)

    def _clear_poly_preview(self):
        for ln in self._preview_lines:
            self.scene.removeItem(ln)
        self._preview_lines.clear()
        if self._preview_close_line is not None:
            self.scene.removeItem(self._preview_close_line)
            self._preview_close_line = None

    def _clamp_scene_point(self, p: QPointF) -> QPointF:
        if self._display_rgb is None:
            return p
        h, w = self._display_rgb.shape[:2]
        x = float(max(0, min(w - 1, p.x())))
        y = float(max(0, min(h - 1, p.y())))
        return QPointF(x, y)

    def _on_toggle_box(self, checked: bool):
        if checked:
            if self.btn_toggle_assign.isChecked():
                self.btn_toggle_assign.blockSignals(True)
                self.btn_toggle_assign.setChecked(False)
                self.btn_toggle_assign.blockSignals(False)
            self.view.set_assign_mode(False)
            self._stop_all_contour_tools()
            self.view.set_box_mode(True, on_complete=self._on_box_drag_finished)
            self.status.setText("Box prompt: drag a rectangle on the image (middle-drag = pan).")
        else:
            self.view.set_box_mode(False)
            self.status.setText("Box prompt off.")

    def _on_box_drag_finished(self, rect: QRectF):
        if not self.btn_toggle_box.isChecked():
            return
        self._run_box_segmentation(rect)

    def _run_box_segmentation(self, rect: QRectF):
        if not self.sam_loaded or self.sam_worker is None:
            QMessageBox.warning(self, "SAM3", "Load the model first.")
            return
        if self._infer_busy:
            self.status.setText("Wait for the current SAM job to finish.")
            return
        if self._display_rgb is None:
            return
        h, w = self._display_rgb.shape[:2]
        r = rect.normalized()
        x1 = max(0.0, min(float(w - 1), r.left()))
        y1 = max(0.0, min(float(h - 1), r.top()))
        x2 = max(0.0, min(float(w - 1), r.right()))
        y2 = max(0.0, min(float(h - 1), r.bottom()))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 4 or (y2 - y1) < 4:
            self.status.setText("Box too small.")
            return

        model_rgb, sx, sy = limit_long_edge(self._display_rgb, self.sp_max_side.value())
        self._last_model_rgb = model_rgb
        self._last_scale = (sx, sy)
        mh, mw = model_rgb.shape[:2]

        xm1 = x1 / sx
        ym1 = y1 / sy
        xm2 = x2 / sx
        ym2 = y2 / sy
        xm1 = max(0.0, min(float(mw - 1), xm1))
        xm2 = max(0.0, min(float(mw - 1), xm2))
        ym1 = max(0.0, min(float(mh - 1), ym1))
        ym2 = max(0.0, min(float(mh - 1), ym2))
        if xm2 <= xm1 or ym2 <= ym1:
            self.status.setText("Invalid box after resize.")
            return

        self.sam_worker._pending_image = model_rgb.copy()
        self.sam_worker._pending_boxes = [[xm1, ym1, xm2, ym2]]
        self.sam_worker._pending_confidence = float(self.sp_conf.value())
        self.sam_worker._pending_max_cells = int(self.sp_max_cells.value())

        self._infer_busy = True
        self.btn_run.setEnabled(False)
        self._set_annotation_tools_enabled(False)
        self.status.setText("Running SAM3 (box)…")
        self.sam_worker.trigger_box_prompt.emit()

    def _set_annotation_tools_enabled(self, enabled: bool):
        """Enable/disable contour tools and SAM prompt buttons (used during SAM inference)."""
        for b in (
            self.btn_freehand,
            self.btn_rubber,
            self.btn_grow,
            self.btn_copy_ann,
            self.btn_paste_ann,
            self.btn_toggle_box,
            self.btn_toggle_assign,
        ):
            b.setEnabled(enabled)

    def _set_sam_prompt_buttons_enabled(self, enabled: bool):
        self.btn_toggle_box.setEnabled(enabled)
        self.btn_toggle_assign.setEnabled(enabled)

    def _selected_annotation_row(self) -> int | None:
        """Annotation index in full per-image list (synced with list widget and contour click)."""
        ei = self._edit.get_selected_contour_index()
        if ei is not None:
            vi = self._edit_ann_indices_valid()
            if 0 <= ei < len(vi):
                return vi[ei]
        row = self.list_annotations.currentRow()
        lst = self._annotations_for_current()
        if 0 <= row < len(lst):
            return row
        return None

    def _delete_selected_annotation(self):
        row = self._selected_annotation_row()
        if row is None:
            self.status.setText("Select a contour to delete (click it or the list), then Delete or click Delete.")
            return
        lst = self._annotations_for_current()
        if row >= len(lst):
            return
        lst.pop(row)
        self._edit.clear_selection(silent=True)
        self.list_annotations.blockSignals(True)
        self.list_annotations.setCurrentRow(-1)
        self.list_annotations.blockSignals(False)
        self._populate_annotation_list_widget()
        self._redraw_annotation_items()
        self.status.setText("Removed annotation.")

    def _save_annotations_json(self):
        path = self._annotations_path or (self._folder / "annotations_tem.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._annotations, f, indent=2)
            self._annotations_path = path
            self.status.setText(f"Saved labels → {path.name}")
        except OSError as e:
            QMessageBox.warning(self, "Save", str(e))

    def _reload_annotations_json(self):
        self._try_load_annotations_file()
        self._populate_annotation_list_widget()
        self._redraw_annotation_items()
        self.status.setText("Reloaded annotations_tem.json")

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "TEM image folder", str(self._folder))
        if d:
            self._folder = Path(d)
            self._refresh_folder_label()
            self._populate_file_list()

    def _populate_file_list(self):
        self.list_w.clear()
        self._files = []
        if not self._folder.is_dir():
            self.status.setText(f"Not a directory: {self._folder}")
            return
        for p in sorted(self._folder.iterdir()):
            if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                self._files.append(p)
                self.list_w.addItem(p.name)
        self._try_load_annotations_file()
        self.status.setText(f"{len(self._files)} image(s).")
        if self._files:
            self.list_w.setCurrentRow(0)

    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self._files):
            return
        self._load_index(row)

    def _load_index(self, idx: int):
        if idx < 0 or idx >= len(self._files):
            return
        self._index = idx
        path = self._files[idx]
        try:
            raw = load_tiff_u16(path)
            if raw.ndim != 2:
                QMessageBox.warning(self, "Load", f"Expected 2D grayscale: {path.name}")
                return
            self._raw_u16 = raw
            self._sam_unlabeled_masks.clear()
            self._overlay_rgb = None
            self._stop_all_contour_tools()
            if self.btn_toggle_assign.isChecked():
                self.btn_toggle_assign.blockSignals(True)
                self.btn_toggle_assign.setChecked(False)
                self.btn_toggle_assign.blockSignals(False)
            self.view.set_assign_mode(False)
            if self.btn_toggle_box.isChecked():
                self.btn_toggle_box.blockSignals(True)
                self.btn_toggle_box.setChecked(False)
                self.btn_toggle_box.blockSignals(False)
            self.view.set_box_mode(False)
            self._refresh_display()
            self._populate_annotation_list_widget()
            self.status.setText(f"Loaded {path.name} ({raw.shape[1]}×{raw.shape[0]})")
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            self.status.setText(str(e))

    def _refresh_display(self):
        if self._raw_u16 is None:
            return
        rgb = u16_to_rgb_u8(
            self._raw_u16,
            p_low=self.sp_p_low.value(),
            p_high=self.sp_p_high.value(),
        )
        self._display_rgb = rgb
        h, w = rgb.shape[:2]
        self._sam_unlabeled_masks = [
            m for m in self._sam_unlabeled_masks if m.shape == (h, w)
        ]
        if self._sam_unlabeled_masks:
            self._overlay_rgb = draw_contours_rgb(
                rgb, self._sam_unlabeled_masks, bgr_color=(0, 255, 0), thickness=2
            )
            show = self._overlay_rgb
        else:
            self._overlay_rgb = None
            show = rgb
        self.pix_item.setPixmap(numpy_rgb_to_qpixmap(show))
        self.scene.setSceneRect(self.pix_item.boundingRect())
        self._redraw_annotation_items()
        self._user_zoomed = False
        self._fit_image_view()

    def _fit_image_view(self):
        if self.pix_item.pixmap() and not self.pix_item.pixmap().isNull():
            self.view.resetTransform()
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
            self._user_zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed and self.pix_item.pixmap() and not self.pix_item.pixmap().isNull():
            self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def _prev(self):
        if self._index > 0:
            self.list_w.setCurrentRow(self._index - 1)

    def _next(self):
        if self._index < len(self._files) - 1:
            self.list_w.setCurrentRow(self._index + 1)

    def _checkpoint_path(self) -> str:
        override = os.environ.get("SAM3_CHECKPOINT")
        if override:
            return override
        return _resolve_sam3_checkpoint_path()

    def _on_load_sam(self):
        if self.sam_loaded:
            QMessageBox.information(self, "SAM3", "Model already loaded.")
            return
        if self.sam_thread and self.sam_thread.isRunning():
            QMessageBox.information(self, "SAM3", "Loading in progress…")
            return

        ckpt = self._checkpoint_path()
        if not os.path.isfile(ckpt):
            tried = (
                "\n".join(_CHECKPOINT_SEARCH_PATHS)
                if not os.environ.get("SAM3_CHECKPOINT")
                else "(SAM3_CHECKPOINT was set)"
            )
            QMessageBox.warning(
                self,
                "Checkpoint",
                f"SAM checkpoint not found:\n{ckpt}\n\n"
                f"Tried:\n{tried}\n\n"
                f"Set SAM3_CHECKPOINT or place sam3.pt next to tem_sam3_viewer.py ({_APP_ROOT}).",
            )
            return

        self.btn_load_sam.setEnabled(False)
        self.btn_load_sam.setText("Loading…")
        self.status.setText("Loading SAM3 (may take a minute)…")

        import torch
        from Models.sam_worker import SamWorker

        force_cpu = os.environ.get("SAM3_FORCE_CPU", "0") == "1"
        force_cuda = os.environ.get("SAM3_FORCE_CUDA", "0") == "1"
        if force_cpu:
            device = "cpu"
        elif force_cuda and torch.cuda.is_available():
            device = "cuda"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        try:
            self.sam_worker = SamWorker(checkpoint_path=ckpt, device=device)
            self.sam_thread = QThread()
            self.sam_worker.moveToThread(self.sam_thread)
        except Exception as e:
            self.btn_load_sam.setEnabled(True)
            self.btn_load_sam.setText("Load SAM3")
            QMessageBox.critical(self, "SAM3", str(e))
            return

        self.sam_thread.started.connect(self.sam_worker.load_model)
        self.sam_worker.predictor_ready.connect(self._on_sam_ready)
        self.sam_worker.error_occurred.connect(self._on_sam_error)
        self.sam_worker.prediction_automatic_complete.connect(self._on_predict_done)
        self.sam_worker.prediction_boxes_complete.connect(self._on_predict_boxes_done)

        self.sam_worker.trigger_automatic.connect(
            self.sam_worker.predict_automatic_async, Qt.QueuedConnection
        )
        self.sam_worker.trigger_box_prompt.connect(
            self.sam_worker.set_image_and_predict_boxes_async, Qt.QueuedConnection
        )

        self.sam_thread.start()

    def _on_sam_ready(self):
        self.sam_loaded = True
        self.btn_load_sam.setText("SAM3 ✓")
        self.btn_load_sam.setEnabled(False)
        self.btn_run.setEnabled(True)
        self._set_sam_prompt_buttons_enabled(True)
        self.status.setText("SAM3 ready — Run SAM or Box.")

    def _on_sam_error(self, msg: str):
        if not self.sam_loaded:
            self.btn_load_sam.setEnabled(True)
            self.btn_load_sam.setText("Load SAM3")
            self.btn_run.setEnabled(False)
            self._set_sam_prompt_buttons_enabled(False)
            QMessageBox.critical(self, "SAM3 load failed", msg)
        else:
            self._infer_busy = False
            self.btn_run.setEnabled(True)
            self._set_annotation_tools_enabled(True)
            QMessageBox.warning(self, "SAM3 inference", msg)
        self.status.setText(msg[:500])

    def _on_run(self):
        if not self.sam_loaded or self.sam_worker is None:
            QMessageBox.warning(self, "SAM3", "Load the model first.")
            return
        if self._infer_busy:
            return
        if self._display_rgb is None:
            QMessageBox.warning(self, "Image", "No image loaded.")
            return

        model_rgb, sx, sy = limit_long_edge(self._display_rgb, self.sp_max_side.value())
        self._last_model_rgb = model_rgb
        self._last_scale = (sx, sy)

        self._infer_busy = True
        self.btn_run.setEnabled(False)
        self._set_annotation_tools_enabled(False)
        self.status.setText("Running SAM3…")

        self.sam_worker._pending_image = model_rgb.copy()
        self.sam_worker._pending_confidence = float(self.sp_conf.value())
        self.sam_worker._pending_max_cells = int(self.sp_max_cells.value())
        self.sam_worker._pending_mask_selection_mode = self.combo_mask_mode.currentData()
        QTimer.singleShot(0, lambda: self.sam_worker.trigger_automatic.emit())

    def _on_predict_done(self, masks):
        self._infer_busy = False
        self.btn_run.setEnabled(True)
        self._set_annotation_tools_enabled(True)
        if self._display_rgb is None:
            self.status.setText("No display image.")
            return
        if masks is None:
            self.status.setText("No masks returned.")
            return
        scaled = self._worker_masks_to_display(normalize_masks(masks))
        self._sam_unlabeled_masks = scaled
        self._rebuild_sam_overlay(
            f"Full-image SAM: {len(scaled)} green contour(s) (replaces previous green SAM). "
            "Box prompts add more; click-assign saves with class."
        )

    def _on_predict_boxes_done(self, masks):
        self._infer_busy = False
        self.btn_run.setEnabled(True)
        self._set_annotation_tools_enabled(True)
        if self._display_rgb is None:
            self.status.setText("No display image.")
            return
        if masks is None:
            self.status.setText("Box: No masks returned.")
            return
        scaled = self._worker_masks_to_display(normalize_masks(masks))
        added, skipped = self._dedupe_append_masks(scaled)
        self._rebuild_sam_overlay(
            f"Box SAM: +{added} new contour(s) ({skipped} duplicate/skipped). "
            f"{len(self._sam_unlabeled_masks)} green total. Turn on “SAM class” to label."
        )

    def _composite_save_rgb(self) -> np.ndarray | None:
        """RGB image: SAM overlay if any, plus rasterized manual contours."""
        if self._display_rgb is None:
            return None
        base = (
            self._overlay_rgb.copy()
            if self._overlay_rgb is not None
            and self._overlay_rgb.shape == self._display_rgb.shape
            else self._display_rgb.copy()
        )
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        for ann in self._annotations_for_current():
            pts = ann.get("contour", [])
            if len(pts) < 3:
                continue
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            col = _color_for_class(str(ann.get("class", "")))
            cv2.polylines(bgr, [arr], True, (col.blue(), col.green(), col.red()), 2)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _on_save_overlay(self):
        if self._display_rgb is None:
            QMessageBox.information(self, "Save", "Load an image first.")
            return
        if self._index < 0 or self._index >= len(self._files):
            return
        comp = self._composite_save_rgb()
        if comp is None:
            return
        default = self._files[self._index].with_suffix(".overlay.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save view (SAM + manual contours)", str(default), "PNG (*.png)"
        )
        if path:
            bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            self.status.setText(f"Saved {path}")

    def closeEvent(self, event):
        if self.sam_thread and self.sam_thread.isRunning():
            self.sam_thread.quit()
            self.sam_thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = TemSam3ViewerWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
