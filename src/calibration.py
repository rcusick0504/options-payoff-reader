"""Validated, user-confirmed axis ticks for charts OCR cannot fully read."""
import math
import numpy as np


def manual_axes(points):
    if not isinstance(points, list) or len(points) > 12:
        raise ValueError("Supply at most twelve confirmed axis ticks.")
    groups = {"x": [], "y": []}
    for point in points:
        if not isinstance(point, dict) or point.get("axis") not in groups:
            raise ValueError("Each confirmed tick must identify the price or vertical axis.")
        position = point.get("position")
        value = point.get("value")
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError("Each confirmed tick needs a position on the image.")
        values = position + [value]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Tick positions and values must be finite numbers.")
        x, y = position
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("Confirmed ticks must lie inside the image.")
        groups[point["axis"]].append(dict(value=float(value), x=x, y=y, right=x, text=str(value), source="user"))
    axes = {}
    for axis, ticks in groups.items():
        if not ticks:
            axes[axis] = None
            continue
        if len(ticks) < 2:
            raise ValueError("Confirm at least two ticks for each axis you correct.")
        coordinates = np.array([t[axis] for t in ticks])
        values = np.array([t["value"] for t in ticks])
        if np.ptp(coordinates) < .05 or len(np.unique(coordinates)) != len(coordinates) or np.ptp(values) == 0:
            raise ValueError("Choose distinct, well-separated ticks with different values.")
        slope, intercept = np.polyfit(coordinates, values, 1)
        error = float(np.max(np.abs(slope * coordinates + intercept - values)) / np.ptp(values))
        if error > .015 or (axis == "x" and slope <= 0) or (axis == "y" and slope >= 0):
            raise ValueError("Confirmed ticks must form a linear scale: prices rise rightward and vertical values rise upward.")
        axes[axis] = dict(slope=float(slope), intercept=float(intercept), ticks=ticks, error=error, source="user")
    return axes
