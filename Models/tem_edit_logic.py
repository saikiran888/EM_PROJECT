"""
Contour editing (freehand, rubber-band vertex sculpt, grow/shrink, selection).
Adapted from AnnotationMasterTool_V3_Dev_SM/Views/edit_panel.py (QCEditLogic).
"""
from typing import Callable, List, Optional, Set, Tuple
import numpy as np
import cv2
import re
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtCore import QObject, pyqtSignal


class TemContourEditLogic(QObject):
    """
    Core editing logic for Contour operations.
    Includes robust 'Click-to-Switch' logic and restores 'select_contour_at'.
    """
    rubberBandToggled = pyqtSignal(bool)
    nudgeRequested = pyqtSignal(int, int)  # dx, dy
    nudgeStepChanged = pyqtSignal(int)
    selectionChanged = pyqtSignal(int, int)
    contoursUpdated = pyqtSignal()
    freehandPreviewUpdated = pyqtSignal(list)  # [(x,y), ...] for live draw

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._nudge_step = 2
        # Influence radius for Grow/Rubber (Image pixels)
        self._radius = 15.0
        # Pixels: treat clicks/hover near the stroke (outside polygon) as hitting the contour
        self._edge_pick_tolerance = 14.0
        
        self._get_contours: Optional[Callable[[], List[np.ndarray]]] = None
        self._set_contour: Optional[Callable[[int, np.ndarray], None]] = None
        self._add_contour: Optional[Callable[[np.ndarray], None]] = None
        self._before_modify_callback: Optional[Callable[[], None]] = None

        self._selected_contour_index: Optional[int] = None
        self._selected_contour_indices: Set[int] = set()
        self._selected_point_indices: Set[int] = set()

        # Mode flags
        self._region_growth_mode: bool = False
        self._rubber_edit_mode: bool = False
        self._freehand_mode: bool = False

        # Editing state
        self._rb_active: bool = False
        self._rb_selected_idx: Optional[int] = None
        self._rb_start_xy: Optional[tuple] = None
        self._rb_original_pts: Optional[np.ndarray] = None

        self._drag_active: bool = False
        self._drag_start_xy: Optional[tuple] = None

        self._drawing: bool = False
        self._freehand_pts: List[tuple] = []
        
        # Copy/paste clipboard
        self._copied_cell: Optional[Tuple[np.ndarray, str]] = None
        
        # Resize state
        self._resize_base_contour: Optional[np.ndarray] = None
        self._resize_slider_last: int = 100

    def set_data_accessors(self, get_contours, set_contour, add_contour=None):
        self._get_contours = get_contours
        self._set_contour = set_contour
        self._add_contour = add_contour

    def set_before_modify_callback(self, cb: Optional[Callable[[], None]]):
        """Called before first contour modification in an edit operation (grow, rubber, drag)."""
        self._before_modify_callback = cb

    def get_selected_contour_index(self) -> Optional[int]:
        return self._selected_contour_index

    def get_selected_contour_indices(self) -> Set[int]:
        """Return set of selected contour indices (multi-select via toolbar or Ctrl+click)."""
        return set(self._selected_contour_indices)

    def is_rubber_edit(self) -> bool:
        """True when Rubber Band sculpt mode is active."""
        return self._rubber_edit_mode

    def is_rubber_band_drag_active(self) -> bool:
        """True when user is actively dragging to sculpt (needs mouse grab for move events)."""
        return self._rubber_edit_mode and self._rb_active

    def is_drag_move_active(self) -> bool:
        """True when user is dragging to move contour."""
        return self._drag_active

    def get_selected_point_indices(self) -> Set[int]:
        return set(self._selected_point_indices)

    def set_nudge_step(self, value: int):
        self._nudge_step = int(value)
        self.nudgeStepChanged.emit(self._nudge_step)
    
    def get_nudge_step(self) -> int:
        return self._nudge_step

    def set_modes(self, region_growth: Optional[bool] = None, rubber_edit: Optional[bool] = None, freehand: Optional[bool] = None):
        if region_growth is not None: self._region_growth_mode = bool(region_growth)
        if rubber_edit is not None: self._rubber_edit_mode = bool(rubber_edit)
        if freehand is not None: self._freehand_mode = bool(freehand)
        self.clear_editing_state()

    def clear_editing_state(self):
        self._rb_active = False
        self._rb_selected_idx = None
        self._rb_original_pts = None
        self._drag_active = False
        self._drawing = False
        self._freehand_pts.clear()

    def clear_selection(self, silent: bool = False):
        """Clear selected contour and point indices. Emits selectionChanged(-1, 0) unless silent."""
        self._selected_contour_index = None
        self._selected_contour_indices.clear()
        self._selected_point_indices.clear()
        if not silent:
            self.selectionChanged.emit(-1, 0)

    # ---- Helpers ----
    def _ensure_contour_array(self, cnt):
        """Unwrap (contour, label, source) or convert to ndarray. Returns numpy contour."""
        if isinstance(cnt, (tuple, list)) and len(cnt) >= 1 and hasattr(cnt[0], 'ndim'):
            return cnt[0]
        if isinstance(cnt, np.ndarray):
            return cnt
        return np.array(cnt, dtype=np.int32)

    def _to_Nx2(self, cnt) -> np.ndarray:
        """Convert contour to (N, 2) array. Handles tuple/list from some data sources."""
        cnt = self._ensure_contour_array(cnt)
        if cnt.ndim == 3:
            return cnt[:, 0, :]
        return cnt

    def _restore_shape(self, original_cnt: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.int32)
        if original_cnt.ndim == 3:
            return pts.reshape((-1, 1, 2))
        return pts

    def _closest_contour_index(self, x: int, y: int) -> Optional[int]:
        """Finds closest contour edge to the point (used when clicking background)."""
        if not self._get_contours: return None
        contours = self._get_contours() or []
        best_idx = None
        best_dist = float('inf')
        pt = (float(x), float(y))
        for i, cnt in enumerate(contours):
            cnt_arr = self._ensure_contour_array(cnt)
            try:
                # abs() distance to edge
                d = abs(cv2.pointPolygonTest(cnt_arr, pt, True))
            except Exception:
                pts = self._to_Nx2(cnt_arr)
                d = np.min(np.linalg.norm(pts - np.array([x, y]), axis=1))
            
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def get_target_contour_index(self, x: int, y: int) -> Optional[int]:
        """Which contour (edit index) is under (x,y); None if outside all contours. Same as pick logic."""
        return self._get_target_contour_index(x, y)

    def _get_target_contour_index(self, x: int, y: int) -> Optional[int]:
        """
        Contour under (x, y): inside the polygon, or within _edge_pick_tolerance px of its edge
        (so clicks on the drawn stroke still hit). Overlapping insides: centroid tie-breaker.
        Outside all: nearest edge within tolerance wins; else None (background click).
        """
        if not self._get_contours:
            return None
        contours = self._get_contours() or []
        if not contours:
            return None

        pt = np.array([float(x), float(y)], dtype=float)
        tol = float(self._edge_pick_tolerance)
        best_idx: Optional[int] = None
        best_dist = -float('inf')
        best_centroid_dist = float('inf')

        for i, cnt in enumerate(contours):
            cnt_arr = self._ensure_contour_array(cnt)
            try:
                dist = float(cv2.pointPolygonTest(cnt_arr, tuple(pt), True))
            except Exception:
                pts = self._to_Nx2(cnt_arr)
                dist = float(-np.min(np.linalg.norm(pts - pt, axis=1)))

            if dist < 0:
                continue

            try:
                M = cv2.moments(cnt_arr)
                if M.get("m00", 0) != 0:
                    c = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=float)
                else:
                    x0, y0, w, h = cv2.boundingRect(cnt_arr)
                    c = np.array([x0 + w / 2.0, y0 + h / 2.0], dtype=float)
            except Exception:
                pts = self._to_Nx2(cnt_arr)
                c = pts.mean(axis=0).astype(float)

            centroid_dist = float(np.linalg.norm(c - pt))

            if (centroid_dist < best_centroid_dist) or (
                abs(centroid_dist - best_centroid_dist) < 1e-6 and dist > best_dist
            ):
                best_centroid_dist = centroid_dist
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            return best_idx

        # Outside all polygons: pick contour whose boundary is closest within tolerance.
        # pointPolygonTest returns -edge_distance when outside.
        near_idx: Optional[int] = None
        near_best = -float('inf')
        for i, cnt in enumerate(contours):
            cnt_arr = self._ensure_contour_array(cnt)
            try:
                dist = float(cv2.pointPolygonTest(cnt_arr, tuple(pt), True))
            except Exception:
                continue
            if -tol < dist < 0 and dist > near_best:
                near_best = dist
                near_idx = i
        return near_idx

    def _apply_gaussian_smoothing(self, pts: np.ndarray, sigma: float = 1.2) -> np.ndarray:
        """
        Light local Gaussian smoothing. Reduces jaggedness without displacing contour.
        Uses mode='wrap' for closed curves. sigma ~1.2 balances smoothness vs fidelity.
        """
        if len(pts) < 5: return pts
        xs = pts[:, 0].astype(float)
        ys = pts[:, 1].astype(float)
        xs_s = gaussian_filter1d(xs, sigma, mode='wrap')
        ys_s = gaussian_filter1d(ys, sigma, mode='wrap')
        return np.column_stack((xs_s, ys_s))

    def _apply_spline_smoothing(self, pts: np.ndarray, s=5.0) -> np.ndarray:
        if len(pts) < 4: return pts
        # Deduplicate
        diffs = np.diff(pts, axis=0)
        keep = np.concatenate(([True], np.linalg.norm(diffs, axis=1) > 0.1))
        pts_clean = pts[keep]
        if len(pts_clean) < 4: return pts

        x = pts_clean[:, 0]
        y = pts_clean[:, 1]
        is_closed = np.linalg.norm(pts_clean[0] - pts_clean[-1]) < 2.0
        try:
            tck, u = splprep([x, y], s=s, per=int(is_closed))
            new_len = len(pts)
            u_new = np.linspace(0, 1, new_len)
            x_new, y_new = splev(u_new, tck)
            return np.column_stack((x_new, y_new))
        except Exception:
            return pts

    def smooth_contour(self, contour: np.ndarray, s: float = 5.0) -> np.ndarray:
        """
        Public helper: apply spline smoothing to a contour (Nx1x2 or Nx2).
        Returns contour in same shape. Used when saving FreeHand to disk.
        """
        pts = self._to_Nx2(contour).astype(float)
        if len(pts) < 4:
            return contour
        smooth_pts = self._apply_spline_smoothing(pts, s=s)
        return self._restore_shape(contour, smooth_pts)

    # ---- Action Implementations ----

    def apply_nudge(self, dx: int, dy: int):
        if self._selected_contour_index is None or not self._get_contours: 
            self.nudgeRequested.emit(dx, dy)
            return
        
        contours = self._get_contours()
        idx = self._selected_contour_index
        if not (0 <= idx < len(contours)): return

        cnt = contours[idx]
        pts = self._to_Nx2(cnt).copy()

        if self._selected_point_indices:
            for i in self._selected_point_indices:
                if 0 <= i < len(pts):
                    pts[i, 0] += dx
                    pts[i, 1] += dy
        else:
            pts[:, 0] += dx
            pts[:, 1] += dy

        self._set_contour(idx, self._restore_shape(cnt, pts))
        self.contoursUpdated.emit()

    def on_rubber_band_complete(self, x1, y1, x2, y2):
        if not self._get_contours: return
        contours = self._get_contours() or []
        
        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])

        if self._selected_contour_index is None:
            found_idx = None
            for idx, cnt in enumerate(contours):
                pts = self._to_Nx2(cnt)
                min_x, max_x = pts[:,0].min(), pts[:,0].max()
                min_y, max_y = pts[:,1].min(), pts[:,1].max()
                if not (xb < min_x or max_x < xa or yb < min_y or max_y < ya):
                    found_idx = idx
                    break
            self._selected_contour_index = found_idx
            self._selected_contour_indices = {found_idx} if found_idx is not None else set()
            self._selected_point_indices.clear()
            self.selectionChanged.emit(found_idx if found_idx is not None else -1, 0)
        else:
            cnt = contours[self._selected_contour_index]
            pts = self._to_Nx2(cnt)
            mask = (pts[:,0] >= xa) & (pts[:,0] <= xb) & (pts[:,1] >= ya) & (pts[:,1] <= yb)
            self._selected_point_indices = set(np.where(mask)[0])
            self.selectionChanged.emit(self._selected_contour_index, len(self._selected_point_indices))

    def _apply_region_growth(self, x: int, y: int):
        if self._selected_contour_index is None or not self._get_contours: return
        contours = self._get_contours()
        idx = self._selected_contour_index
        if not (0 <= idx < len(contours)): return

        cnt = contours[idx]
        pts = self._to_Nx2(cnt).astype(float)
        click_pos = np.array([x, y])

        dists = np.linalg.norm(pts - click_pos, axis=1)
        closest_idx = np.argmin(dists)
        closest_pt = pts[closest_idx]
        
        raw_displacement = click_pos - closest_pt 
        mag = np.linalg.norm(raw_displacement)
        if mag > 0:
            scale = min(mag, 15.0) / mag # Clamp speed
            displacement = raw_displacement * scale
        else:
            displacement = raw_displacement

        # Linear Falloff - include all points within R (including closest at d=0)
        R = self._radius
        d_to_click = np.linalg.norm(pts - click_pos, axis=1)
        mask = d_to_click <= R
        # Avoid division by zero when R is 0; skip if no points in range
        if R <= 0:
            return

        modified = False
        if np.any(mask):
            weights = (R - d_to_click[mask]) / R
            pts[mask] += displacement * weights[:, np.newaxis]
            modified = True

        if modified:
            if self._before_modify_callback:
                self._before_modify_callback()
            # Light Gaussian smoothing: reduces jaggedness without drift (unlike spline)
            pts = self._apply_gaussian_smoothing(pts, sigma=1.2)
            self._set_contour(idx, self._restore_shape(cnt, pts))
            self.contoursUpdated.emit()

    # ---- Mouse Events ----
    def on_mouse_press(self, button: int, x: int, y: int, shift_modifier: bool = False):
        """
        Handles mouse press.
        shift_modifier: when True, add/remove contour from multi-selection instead of single select
        (toolbar Multi-select or Ctrl+click from the view).
        """
        # 1. Identify what is strictly 'under' the cursor for selection purposes
        target_idx = self._get_target_contour_index(x, y)

        if button == 1:  # Left Click
            # A. Freehand Mode
            if self._freehand_mode:
                self._drawing = True
                self._freehand_pts = [(int(x), int(y))]
                self.freehandPreviewUpdated.emit(list(self._freehand_pts))
                return

            # B. RUBBER BAND SCULPTING (Prioritize if a cell is already selected)
            # We do this BEFORE selection switching to allow pulling edges outward.
            if self._rubber_edit_mode and self._selected_contour_index is not None:
                contours = self._get_contours()
                if 0 <= self._selected_contour_index < len(contours):
                    cnt = contours[self._selected_contour_index]
                    pts = self._to_Nx2(cnt)
                    
                    # Distance from mouse (x,y) to every point on the contour line
                    dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
                    closest_i = int(np.argmin(dists))
                    
                    # Generous radius: 100 image pixels. 
                    # If this is still too small, you can remove the 'if' entirely 
                    # to grab the nearest point anywhere on the screen.
                    if dists[closest_i] < 100.0: 
                        if self._before_modify_callback:
                            self._before_modify_callback()
                        self._rb_selected_idx = closest_i
                        self._rb_start_xy = (int(x), int(y))
                        self._rb_original_pts = pts.copy()
                        self._rb_active = True
                        return # Successfully grabbed a point, stop here.

            # C. Selection Switch (multi-select: add/remove from set)
            if target_idx is not None:
                if shift_modifier:
                    if target_idx in self._selected_contour_indices:
                        self._selected_contour_indices.discard(target_idx)
                    else:
                        self._selected_contour_indices.add(target_idx)
                    self._selected_point_indices.clear()
                    if self._selected_contour_indices:
                        primary = min(self._selected_contour_indices)
                        self._selected_contour_index = primary
                    else:
                        self._selected_contour_index = None
                    self.selectionChanged.emit(
                        self._selected_contour_index if self._selected_contour_index is not None else -1,
                        len(self._selected_contour_indices),
                    )
                    return
                elif target_idx != self._selected_contour_index:
                    self._selected_contour_index = target_idx
                    self._selected_contour_indices = {target_idx}
                    self._selected_point_indices.clear()
                    self.selectionChanged.emit(target_idx, 0)
                    # Same gesture as right-click: select and immediately allow drag.
                    if self._before_modify_callback:
                        self._before_modify_callback()
                    self._drag_start_xy = (int(x), int(y))
                    self._drag_active = True
                    return

            # D. Grow/Shrink Mode
            if self._region_growth_mode and self._selected_contour_index is not None:
                self._apply_region_growth(x, y)
                return

            # E. Deselection
            # (Clicking pure background while no edit tool is active)
            if target_idx is None and not (self._region_growth_mode or self._rubber_edit_mode):
                if shift_modifier:
                    pass  # Keep selection when multi-select (Ctrl+click / toolbar) and click background
                else:
                    self._selected_contour_index = None
                    self._selected_contour_indices.clear()
                    self._selected_point_indices.clear()
                    self.selectionChanged.emit(-1, 0)
                return

            # F. Drag-to-Move Fallback (single contour only)
            if target_idx is not None and target_idx == self._selected_contour_index:
                if self._before_modify_callback:
                    self._before_modify_callback()
                self._drag_start_xy = (int(x), int(y))
                self._drag_active = True
                return

        elif button == 2:  # Right Click
            self._selected_contour_index = target_idx
            self._selected_contour_indices = {target_idx} if target_idx is not None else set()
            self._selected_point_indices.clear()
            self.selectionChanged.emit(target_idx if target_idx is not None else -1, 0)
            if target_idx is not None:
                if self._before_modify_callback:
                    self._before_modify_callback()
                self._drag_start_xy = (int(x), int(y))
                self._drag_active = True


    def on_mouse_move(self, x: int, y: int):
        # Rubber Band
        if self._rubber_edit_mode and self._rb_active:
            contours = self._get_contours()
            idx = self._selected_contour_index
            if 0 <= idx < len(contours):
                cnt = contours[idx]
                pts = self._rb_original_pts.astype(float)
                
                dx = x - self._rb_start_xy[0]
                dy = y - self._rb_start_xy[1]
                
                N = len(pts)
                R_idx = 10 
                center_idx = self._rb_selected_idx
                indices = np.arange(N)
                dist_idx = np.abs(indices - center_idx)
                dist_idx = np.minimum(dist_idx, N - dist_idx)
                
                weights = np.maximum(0, (R_idx - dist_idx) / R_idx)
                pts[:, 0] += dx * weights
                pts[:, 1] += dy * weights
                # Gaussian smoothing: reduces jaggedness without drift (unlike spline)
                pts = self._apply_gaussian_smoothing(pts, sigma=1.2)
                self._set_contour(idx, self._restore_shape(cnt, pts))
                self.contoursUpdated.emit()
            return

        # Move Contour
        if self._drag_active and self._selected_contour_index is not None:
            contours = self._get_contours()
            idx = self._selected_contour_index
            if 0 <= idx < len(contours):
                cnt = contours[idx]
                pts = self._to_Nx2(cnt).copy()
                dx = int(x) - int(self._drag_start_xy[0])
                dy = int(y) - int(self._drag_start_xy[1])
                pts[:, 0] += dx
                pts[:, 1] += dy
                self._set_contour(idx, self._restore_shape(cnt, pts))
                self.contoursUpdated.emit()
                self._drag_start_xy = (int(x), int(y))
            return

        # Freehand: append point and emit preview for live drawing
        if self._freehand_mode and self._drawing:
            self._freehand_pts.append((int(x), int(y)))
            self.freehandPreviewUpdated.emit(list(self._freehand_pts))
            return

    def on_mouse_release(self, button: int, x: int, y: int):
        self._rb_active = False
        self._drag_active = False
        
        if self._freehand_mode and self._drawing and button == 1:
            self._drawing = False
            if len(self._freehand_pts) > 2:
                pts = np.array(self._freehand_pts, dtype=np.int32).reshape((-1, 1, 2))
                
                flat_pts = pts.reshape((-1, 2)).astype(float)
                smooth_pts = self._apply_spline_smoothing(flat_pts, s=10.0)
                final_pts = self._restore_shape(pts, smooth_pts)
                
                if self._add_contour:
                    self._add_contour(final_pts)
                self.contoursUpdated.emit()
            self._freehand_pts.clear()

    def on_hover(self, x: int, y: int):
        """
        Navigate mode: moving over a contour selects it (hover), so outline updates without a click.
        Does not clear selection when the cursor moves to empty space (only a background click does).
        Skipped while dragging, rubber sculpt drag, or draw modes.
        """
        if self._drag_active or self._rb_active or self._drawing:
            return
        if self._freehand_mode or self._region_growth_mode:
            return
        if self._rubber_edit_mode:
            return
        if len(self._selected_contour_indices) > 1:
            return
        target_idx = self._get_target_contour_index(int(x), int(y))
        if target_idx is None:
            return
        if target_idx == self._selected_contour_index and self._selected_contour_indices == {target_idx}:
            return
        self._selected_contour_index = target_idx
        self._selected_contour_indices = {target_idx}
        self._selected_point_indices.clear()
        self.selectionChanged.emit(target_idx, 0)

    # ---- Public Selection Helper (RESTORED) ----
    def select_contour_at(self, x: int, y: int):
        """
        Public method to allow View to force selection.
        Uses the shared _get_target_contour_index logic.
        """
        idx = self._get_target_contour_index(x, y)
        self._selected_contour_index = idx
        self._selected_contour_indices = {idx} if idx is not None else set()
        self._selected_point_indices.clear()
        self.selectionChanged.emit(idx if idx is not None else -1, 0)

    # ---- Label Management ----
    def get_next_label_for_prefix(
        self, prefix: str, contour_info: List[Tuple] = None, selected_idx: int = None
    ) -> str:
        """
        Return next available label for prefix (e.g., M_1, E_2, T_3).

        The numeric suffix is unique across all class labels in this frame (any
        prefix): e.g. after T_1 and E_2, the next E is E_3, not E_1.

        Args:
            prefix: Label prefix (M, F, B, E, T, Y)
            contour_info: List of (contour, label, source) tuples
            selected_idx: Index of currently selected contour (excluded when
                counting used numbers so replacing a label frees its old id)

        Returns:
            Label string like "M_1", "F_2", etc.
        """
        if not contour_info:
            return f"{prefix}_1"

        prefix_pat = re.compile(r"^" + re.escape(prefix) + r"_(\d+)$", re.IGNORECASE)
        # Any Letter_digits label (T_1, E_2, ...) shares one global counter per frame.
        global_suffix_pat = re.compile(r"^[A-Za-z]+_(\d+)$", re.IGNORECASE)

        # If labeling an existing selected cell, keep its label if it already has this prefix
        if selected_idx is not None and 0 <= selected_idx < len(contour_info):
            item = contour_info[selected_idx]
            if len(item) >= 2:
                label = str(item[1]).strip()
                m = prefix_pat.match(label)
                if m:
                    return label  # Already has prefix_n, keep it

        used_global: Set[int] = set()
        for i, item in enumerate(contour_info):
            if selected_idx is not None and i == selected_idx:
                continue  # replaced label does not reserve a number
            if len(item) < 2:
                continue
            label = str(item[1]).strip()
            m = global_suffix_pat.match(label)
            if m:
                used_global.add(int(m.group(1)))

        n = 1
        while n in used_global:
            n += 1
        return f"{prefix}_{n}"

    def assign_label_to_contour(
        self, prefix: str, contour_info: List[Tuple], selected_idx: int = None
    ) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
        """
        Assign a label to the selected contour.
        
        Args:
            prefix: Label prefix (M, F, B, E, T, Y)
            contour_info: List of (contour, label, source) tuples
            selected_idx: Index to label (defaults to current selection)
            
        Returns:
            Tuple of (success, updated_contour_info, new_label)
        """
        if selected_idx is None:
            selected_idx = self._selected_contour_index
        
        if selected_idx is None or not contour_info or selected_idx < 0 or selected_idx >= len(contour_info):
            return False, None, None
        
        label = self.get_next_label_for_prefix(prefix, contour_info=contour_info, selected_idx=selected_idx)
        
        # Update the label
        info = list(contour_info)
        item = list(info[selected_idx])
        item[1] = label
        info[selected_idx] = tuple(item)
        
        return True, info, label

    def set_contour_label_string(
        self, contour_info: List[Tuple], new_label: str, selected_idx: int = None
    ) -> Tuple[bool, Optional[List[Tuple]], Optional[str]]:
        """Set the selected contour's label string (e.g. T_2) without prefix auto-pick."""
        if selected_idx is None:
            selected_idx = self._selected_contour_index
        if selected_idx is None or not contour_info or selected_idx < 0 or selected_idx >= len(contour_info):
            return False, None, None
        info = list(contour_info)
        item = list(info[selected_idx])
        item[1] = (new_label or "").strip()
        info[selected_idx] = tuple(item)
        return True, info, item[1]

    # ---- Copy/Paste Operations ----
    def copy_selected_contour(self, contour_info: List[Tuple]) -> bool:
        """
        Copy the selected contour to clipboard.
        
        Args:
            contour_info: List of (contour, label, source) tuples
            
        Returns:
            True if copied successfully
        """
        idx = self._selected_contour_index
        if idx is None or not contour_info or idx < 0 or idx >= len(contour_info):
            return False
        
        item = contour_info[idx]
        contour = item[0]
        label = item[1] if len(item) >= 2 else "cell"
        
        self._copied_cell = (contour.copy(), label)
        return True

    def paste_contour(self, contour_info: List[Tuple]) -> Tuple[bool, Optional[List[Tuple]]]:
        """
        Paste the copied contour.
        
        Args:
            contour_info: Current list of (contour, label, source) tuples
            
        Returns:
            Tuple of (success, updated_contour_info)
        """
        if not self._copied_cell:
            return False, None
        
        contour_copy, label = self._copied_cell
        if contour_copy is None:
            return False, None
        
        # Add copied contour to the list
        info = list(contour_info) if contour_info else []
        info.append((contour_copy.copy(), label, label))
        
        return True, info

    def has_copied_contour(self) -> bool:
        """Check if there's a contour in the clipboard."""
        return self._copied_cell is not None

    # ---- Resize Operations ----
    def start_resize(self, contour: np.ndarray):
        """
        Start a resize operation by storing the base contour.
        
        Args:
            contour: The contour to resize
        """
        self._resize_base_contour = contour.copy() if contour is not None else None
        self._resize_slider_last = 100

    def apply_resize(self, scale_percent: int, push_undo: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Apply resize to the base contour.
        
        Args:
            scale_percent: Scale percentage (100 = original, 150 = 1.5x, 50 = 0.5x)
            push_undo: Whether this is the first change (for undo tracking)
            
        Returns:
            Tuple of (success, resized_contour)
        """
        if self._resize_base_contour is None or self._resize_base_contour.size == 0:
            return False, None
        
        base_cnt = self._resize_base_contour
        base_pts = base_cnt[:, 0, :].astype(np.float64) if base_cnt.ndim == 3 else base_cnt.astype(np.float64)
        
        # Calculate centroid
        centroid = np.mean(base_pts, axis=0)
        
        # Apply scale
        scale = scale_percent / 100.0
        new_pts = centroid + scale * (base_pts - centroid)
        new_pts = new_pts.astype(np.int32)
        
        # Restore original shape
        if base_cnt.ndim == 3:
            new_pts = new_pts.reshape(-1, 1, 2)
        
        self._resize_slider_last = scale_percent
        return True, new_pts

    def reset_resize(self):
        """Reset resize state."""
        self._resize_base_contour = None
        self._resize_slider_last = 100

    def get_resize_slider_value(self) -> int:
        """Get the last resize slider value."""
        return self._resize_slider_last

    def set_selected_contour_index(self, idx: Optional[int], *, silent: bool = False):
        """Programmatically select an annotation contour by index (e.g. from list widget)."""
        self._selected_point_indices.clear()
        if idx is None or idx < 0:
            self._selected_contour_index = None
            self._selected_contour_indices.clear()
            if not silent:
                self.selectionChanged.emit(-1, 0)
            return
        self._selected_contour_index = int(idx)
        self._selected_contour_indices = {int(idx)}
        if not silent:
            self.selectionChanged.emit(int(idx), 0)