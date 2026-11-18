"""
#Rally analysis (runs after detection, on the stored per-frame results)

- split the birdie track into rallies
- find hits (sharp direction change next to a player)
  * landing position of each rally (last seen birdie point -> court meters)
  * shot speed and shot type (smash / clear / drop / net / lift / drive)

Why landings & shot ends use the GROUND plane only:
the homography maps the floor, so it is only valid where the birdie touches the
floor (landing) or at a player's feet (hit / receive position). Mid-air birdie
pixels are never pushed through the homography.
"""
import numpy as np

MPS_TO_MPH = 2.23694

# Badminton heuristics (meters from the net, m/s, arc = height / court pixel length).
# Tune these on your annotated validation videos.
SHOT_RULES = {
    "net_zone": 2.2,         # within this distance of the net = front court
    "rear_zone": 4.2,        # beyond this distance of the net = rear court
    "smash_speed": 16.0,     # avg horizontal speed (m/s) for a smash
    "drive_speed": 9.0,
    "flat_arc": 0.12,        # arc below this = flat trajectory
    "high_arc": 0.30,        # arc above this = high trajectory
}


def segment_rallies(birdie, fps, gap_s=1.0, min_len_s=0.6):
    """birdie: list (per frame) of (x, y, status) or None. Returns [(start, end)]."""
    gap, min_len = int(gap_s * fps), int(min_len_s * fps)
    rallies, start, last_seen = [], None, None
    for f, b in enumerate(birdie):
        if b is None or b[2] != "seen":
            continue
        if start is None:
            start = f
        elif f - last_seen > gap:
            rallies.append((start, last_seen))
            start = f
        last_seen = f
    if start is not None:
        rallies.append((start, last_seen))
    return [(s, e) for s, e in rallies if e - s >= min_len]


def _trajectory(birdie, s, e):
    #Interpolated, lightly smoothed (N, 2) image trajectory for frames 
    frames = np.arange(s, e + 1)
    # only real detections; Kalman predictions overshoot exactly at hits
    known = [(f, birdie[f][0], birdie[f][1]) for f in frames
             if birdie[f] is not None and birdie[f][2] == "seen"]
    kf = np.array([k[0] for k in known])
    xs = np.interp(frames, kf, [k[1] for k in known])
    ys = np.interp(frames, kf, [k[2] for k in known])
    traj = np.stack([xs, ys], axis=1)
    seen = np.array([birdie[f] is not None and birdie[f][2] == "seen" for f in frames])
    if len(traj) >= 5:
        kernel = np.ones(3) / 3
        traj[1:-1, 0] = np.convolve(traj[:, 0], kernel, "valid")
        traj[1:-1, 1] = np.convolve(traj[:, 1], kernel, "valid")
    return frames, traj, seen


def _gap_runs(seen):
    runs, start = [], None
    for i, s in enumerate(seen):
        if not s and start is None:
            start = i
        elif s and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(seen) - 1))
    return runs


def _dist_to_reach(pt, box):
    #Pixel distance from the birdie to a player's racket-reach area (0 if inside)
    x, y, w, h = box
    x0, x1 = x - 0.6 * h, x + w + 0.6 * h   # arm + racket to the sides
    y0, y1 = y - 0.5 * h, y + h    # and above the head
    dx = max(x0 - pt[0], 0, pt[0] - x1)
    dy = max(y0 - pt[1], 0, pt[1] - y1)
    return float(np.hypot(dx, dy))


def _nearest_player(pt, players_at_frame, side=None, court=None):
    #Closest player by racket reach (ties broken by distance to the head).
    #If side is given, prefer players on that half of the court
    best, best_key, best_d = None, (float("inf"), float("inf")), float("inf")
    for p in players_at_frame or []:
        if side and court and court.side(p["court"]) != side:
            continue
        x, y, w, h = p["box"]
        d = _dist_to_reach(pt, p["box"])
        key = (d, np.hypot(pt[0] - (x + w / 2), pt[1] - y) / max(h, 1))
        if key < best_key:
            best, best_key, best_d = p, key, d
    if best is None and side:
        return _nearest_player(pt, players_at_frame)
    return best, best_d


def detect_hits(frames, traj, seen, players, fps, min_angle=50.0):
    k = max(2, int(round(0.05 * fps)))
    min_sep = int(0.25 * fps)
    scores = np.zeros(len(traj))
    for i in range(k, len(traj) - k):
        v_in = traj[i] - traj[i - k]
        v_out = traj[i + k] - traj[i]
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in < 1e-3 or n_out < 1e-3:
            continue
        ang = np.degrees(np.arccos(np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1, 1)))
        accel = n_out / n_in
        if not seen[max(0, i - k):i + k + 1].all():
            # corner bridged by interpolation: the top of a smooth arc hidden behind a
            # player also looks like a corner, so demand a real reversal
            if ang < 100:
                continue
        elif ang < min_angle and accel < 2.5:
            continue
        p, d = _nearest_player(traj[i], players.get(int(frames[i])))
        if p is None or d > 0:          # a hit has to happen within racket reach
            continue
        scores[i] = ang + 30 * min(accel, 4)

    hits = []
    for i in np.argsort(-scores):
        if scores[i] <= 0:
            break
        if all(abs(i - j) > min_sep for j in hits):
            hits.append(int(i))
    # A birdie is often hidden by the hitter's body at the moment of contact, so the
    # corner shows up where tracking resumes. Move such hits to the middle of the gap.
    runs = [g for g in _gap_runs(seen) if g[1] - g[0] >= 2]
    moved = []
    for i in hits:
        near = [g for g in runs if g[0] - k <= i <= g[1] + k + 1]
        moved.append((near[0][0] + near[0][1]) // 2 if near else i)
    hits = sorted(set(moved))

    # the serve / first contact is usually hidden by the server's body, so the rally's
    # first tracked point starts a shot without a detected hit
    serve_added = not hits or hits[0] > min_sep
    if serve_added:
        hits.insert(0, 0)
    return hits, serve_added


def _arc(points, court):
    #How far the flight bulges ABOVE the straight line between its ends
    a, b = points[0], points[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-6 or len(points) < 3:
        return 0.0
    normal = np.array([ab[1], -ab[0]]) / n
    if normal[1] > 0:                    # make the normal point UP in the image
        normal = -normal
    heights = (points - a) @ normal
    return float(max(0.0, heights.max()) / max(court.image_length_px, 1.0))


def classify_shot(start_d, end_d, speed, arc, rules=SHOT_RULES):
    #start_d / end_d are distances from the net in meters
    if start_d <= rules["net_zone"] and end_d <= rules["net_zone"] + 0.5:
        return "net shot"
    if start_d >= rules["rear_zone"] - 0.7:
        if speed >= rules["smash_speed"] and arc <= rules["high_arc"]:
            return "smash"
        if end_d >= rules["rear_zone"] and arc >= rules["high_arc"]:
            return "clear"
        if end_d <= rules["net_zone"] + 0.5:
            return "drop"
    if start_d <= rules["net_zone"] + 1.0 and end_d >= rules["rear_zone"] and arc >= rules["flat_arc"]:
        return "lift"
    if arc <= rules["flat_arc"] and speed >= rules["drive_speed"]:
        return "drive"
    return "other"


def analyse(birdie, players, court, fps, game="doubles", classify=True, frame_h=None):
    #birdie: per-frame list; players: {frame: [ {id, court, box}, ... ]}
    rallies_out, shots_out = [], []
    for r_idx, (s, e) in enumerate(segment_rallies(birdie, fps), start=1):
        frames, traj, seen = _trajectory(birdie, s, e)
        hits, serve_added = detect_hits(frames, traj, seen, players, fps)

        #landing (end of rally)
        land_img = traj[-1]
        land_court = court.to_court([land_img])[0]
        tail = traj[-max(3, int(0.1 * fps)):]
        falling = (tail[-1][1] - tail[0][1]) > 0        # moving down in the image
        landed_in = court.is_in(land_court, game)
        land_side = court.side(land_court)

        #shots between hits 
        # who hit each shot
        # Badminton hits alternate between the halves. Anchor on the hit where the
        # birdie was closest to a racket and alternate from there; this avoids the
        # perspective trap where a high birdie *looks* close to the far player.
        reach = []
        for h in hits:
            p, d = _nearest_player(traj[h], players.get(int(frames[h])))
            reach.append((d, court.side(p["court"]) if p else None))
        other = {"near": "far", "far": "near"}
        sides = [None] * len(hits)
        valid = [i for i, (d, sd) in enumerate(reach)
                 if sd is not None and not (i == 0 and serve_added)]
        if valid:
            a = min(valid, key=lambda i: reach[i][0])
            anchor_side = reach[a][1]
        else:
            # only the serve: the server is on the half opposite to where it landed
            a, anchor_side = len(hits) - 1, other[land_side] if landed_in else None
        if anchor_side:
            for i in range(len(hits)):
                sides[i] = anchor_side if (i - a) % 2 == 0 else other[anchor_side]

        # shots between hits 
        rally_shots = []
        for h_i, h in enumerate(hits):
            end_i = hits[h_i + 1] if h_i + 1 < len(hits) else len(traj) - 1
            if end_i - h < 2:
                continue
            f0, f1 = int(frames[h]), int(frames[end_i])
            hitter, _ = _nearest_player(traj[h], players.get(f0), sides[h_i], court)
            if hitter is None:
                continue
            start_c = np.asarray(hitter["court"], float)
            is_last = h_i + 1 >= len(hits)
            if is_last:
                end_c = np.asarray(land_court, float)
            else:
                receiver, _ = _nearest_player(traj[end_i], players.get(f1), sides[h_i + 1], court)
                end_c = np.asarray(receiver["court"] if receiver else land_court, float)

            dur = (f1 - f0) / fps
            dist = float(np.linalg.norm(end_c - start_c))
            speed = dist / dur if dur > 0 else 0.0
            arc = _arc(traj[h:end_i + 1], court)
            if frame_h and any(birdie[f] is not None and birdie[f][1] < 0.04 * frame_h
                               for f in range(f0, f1 + 1)):
                arc = max(arc, 1.0)      # went out of the top of the frame = very high
            start_d, end_d = court.dist_from_net(start_c), court.dist_from_net(end_c)
            shot_type = classify_shot(start_d, end_d, speed, arc) if classify else "shot"

            rally_shots.append({
                "rally": r_idx, "shot": len(rally_shots) + 1,
                "start_frame": f0, "end_frame": f1,
                "hitter_id": hitter["id"], "hitter_side": court.side(start_c),
                "start_x_m": round(float(start_c[0]), 2), "start_y_m": round(float(start_c[1]), 2),
                "end_x_m": round(float(end_c[0]), 2), "end_y_m": round(float(end_c[1]), 2),
                "duration_s": round(dur, 3), "distance_m": round(dist, 2),
                "avg_speed_mps": round(speed, 2), "avg_speed_mph": round(speed * MPS_TO_MPH, 1),
                "arc": round(arc, 3), "type": shot_type, "ends_rally": is_last,
            })

        last = rally_shots[-1] if rally_shots else None
        if landed_in:
            point_to = "near" if land_side == "far" else "far"
        else:
            point_to = ("far" if last["hitter_side"] == "near" else "near") if last else None

        rallies_out.append({
            "rally": r_idx, "start_frame": s, "end_frame": e,
            "start_time_s": round(s / fps, 2), "duration_s": round((e - s) / fps, 2),
            "num_shots": len(rally_shots),
            "landing_x_m": round(float(land_court[0]), 2),
            "landing_y_m": round(float(land_court[1]), 2),
            "landing_side": land_side, "landing_in": bool(landed_in),
            "landing_confidence": "high" if falling else "low",
            "final_shot_type": last["type"] if last else None,
            "final_shot_speed_mph": last["avg_speed_mph"] if last else None,
            "point_to_side": point_to,
        })
        shots_out += rally_shots
    return rallies_out, shots_out
