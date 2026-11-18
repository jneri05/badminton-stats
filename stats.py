
#Player movement stats, heatmaps and the 2D court map (the proposal's final output).

import csv
import json
from collections import Counter

import cv2
import numpy as np

from rally import MPS_TO_MPH

PLAYER_COLORS = [(0, 200, 255), (255, 120, 0), (0, 255, 120), (255, 0, 200),
                 (120, 120, 255), (255, 255, 0)]


def player_color(pid):
    return PLAYER_COLORS[(pid - 1) % len(PLAYER_COLORS)]


def player_stats(tracks, court, fps, window_s=0.25, max_speed=9.0):
    #tracks: [(id, [(frame, x, y, coasted), ...])] in court meters
    out = {}
    w = max(1, int(window_s * fps))
    for pid, hist in tracks:
        pts = np.array([[h[1], h[2]] for h in hist])
        frames = np.array([h[0] for h in hist])
        speeds = []
        for i in range(w, len(pts)):
            dt = (frames[i] - frames[i - w]) / fps
            if dt <= 0:
                continue
            v = np.linalg.norm(pts[i] - pts[i - w]) / dt
            if v <= max_speed:    # drop tracking glitches
                speeds.append(v)
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        distance = float(step[step < max_speed / fps * 3].sum())

        zones = Counter(court.zone(p) for p in pts)
        total = sum(zones.values())
        zone_pct = {z: round(100 * c / total, 1) for z, c in zones.most_common()}
        sides = Counter(court.side(p) for p in pts)

        out[pid] = {
            "side": sides.most_common(1)[0][0],
            "time_tracked_s": round(len(pts) / fps, 1),
            "distance_m": round(distance, 1),
            "avg_speed_mps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "max_speed_mps": round(float(np.percentile(speeds, 98)), 2) if speeds else 0.0,
            "avg_speed_mph": round(float(np.mean(speeds)) * MPS_TO_MPH, 1) if speeds else 0.0,
            "max_speed_mph": round(float(np.percentile(speeds, 98)) * MPS_TO_MPH, 1) if speeds else 0.0,
            "most_time_zone": next(iter(zone_pct), None),
            "zone_time_pct": zone_pct,
        }
    return out


def heatmap_grid(points, court, cell=0.1, margin=1.0):
    gw = int((court.width + 2 * margin) / cell)
    gh = int((court.length + 2 * margin) / cell)
    grid = np.zeros((gh, gw), np.float32)
    for x, y in points:
        gx, gy = int((x + margin) / cell), int((y + margin) / cell)
        if 0 <= gx < gw and 0 <= gy < gh:
            grid[gy, gx] += 1
    return cv2.GaussianBlur(grid, (0, 0), sigmaX=3.0)


def _overlay_heat(canvas, grid, alpha=0.6):
    if grid.max() <= 0:
        return canvas
    norm = (grid / grid.max() * 255).astype(np.uint8)
    norm = cv2.resize(norm, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mask = (norm > 15)[..., None]
    blended = cv2.addWeighted(canvas, 1 - alpha, color, alpha, 0)
    return np.where(mask, blended, canvas)


def render_court_map(court, tracks, rallies, px_per_m=50, margin=1.0, title="All players"):
    #Heatmap of every player + landing dots labelled with speed (mph)."""
    canvas, to_px = court.map_canvas(px_per_m, margin)
    pts = [(h[1], h[2]) for _, hist in tracks for h in hist]
    canvas = _overlay_heat(canvas, heatmap_grid(pts, court, margin=margin))
    for a, b in court.lines:                       # redraw lines on top of the heat
        cv2.line(canvas, to_px(a), to_px(b), (255, 255, 255), 2, cv2.LINE_AA)

    for r in rallies:
        p = to_px((r["landing_x_m"], r["landing_y_m"]))
        col = (0, 255, 0) if r["landing_in"] else (0, 0, 255)
        cv2.circle(canvas, p, 7, col, -1)
        cv2.circle(canvas, p, 7, (0, 0, 0), 1)
        label = f"{r['final_shot_speed_mph']:.0f} mph" if r.get("final_shot_speed_mph") else f"R{r['rally']}"
        cv2.putText(canvas, label, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (p[0] + 9, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    header = np.full((40, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(header, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend = np.full((30, canvas.shape[1], 3), 30, np.uint8)
    cv2.circle(legend, (15, 15), 6, (0, 255, 0), -1)
    cv2.putText(legend, "in", (26, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(legend, (60, 15), 6, (0, 0, 255), -1)
    cv2.putText(legend, "out  (label = final shot speed)", (71, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([header, canvas, legend])


def render_player_heatmap(court, pid, hist, px_per_m=50, margin=1.0):
    canvas, to_px = court.map_canvas(px_per_m, margin)
    canvas = _overlay_heat(canvas, heatmap_grid([(h[1], h[2]) for h in hist], court, margin=margin))
    for a, b in court.lines:
        cv2.line(canvas, to_px(a), to_px(b), (255, 255, 255), 2, cv2.LINE_AA)
    header = np.full((40, canvas.shape[1], 3), 30, np.uint8)
    cv2.putText(header, f"Player {pid}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                player_color(pid), 2)
    return np.vstack([header, canvas])


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)
