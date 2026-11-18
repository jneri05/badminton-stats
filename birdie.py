"""
Birdie detection + tracking.

Candidates can come from three sources, all fed into the same Kalman tracker so
the classical and deep approaches can be compared fairly (proposal: classical vs deep):
- MotionBirdieDetector: 3-frame differencing, small bright blobs
- YOLO custom class: a fine-tuned YOLO with a "shuttlecock" or "birdie" class
- TrackNet CSV: output of TrackNet (Frame, Visibility, X, Y)
"""
import csv

import cv2
import numpy as np

BIRDIE_LABELS = ("shuttlecock", "birdie", "shuttle")


class MotionBirdieDetector:
    def __init__(self, frame_shape, diff_threshold=25):
        h, w = frame_shape[:2]
        scale = (w / 1920.0) ** 2            # blob size depends on resolution
        self.min_area = max(1.0, 3 * scale)
        self.max_area = max(30.0, 400 * scale)
        self.diff_threshold = diff_threshold
        self.prev1 = None
        self.prev2 = None
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame, player_boxes=()):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        cands = []
        if self.prev1 is not None and self.prev2 is not None:
            # pixels that differ from BOTH previous frames = object's *current* position
            d1 = cv2.absdiff(gray, self.prev1)
            d2 = cv2.absdiff(gray, self.prev2)
            motion = cv2.bitwise_and(d1, d2)
            _, mask = cv2.threshold(motion, self.diff_threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, self.kernel, iterations=1)

            n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if not (self.min_area <= area <= self.max_area):
                    continue
                if max(bw, bh) > 6 * max(1, min(bw, bh)):      # long streaks = lines/rackets
                    continue
                cx, cy = cents[i]
                brightness = float(gray[int(cy), int(cx)])
                cands.append((float(cx), float(cy), brightness, _inside_body(cx, cy, player_boxes)))
        self.prev2, self.prev1 = self.prev1, gray
        return cands


def _inside_body(x, y, boxes, pad=0.1):
    #capturing the players head, arm, racket possibly
    for bx, by, bw, bh in boxes:
        if (bx - pad * bw <= x <= bx + (1 + pad) * bw and
                by - pad * bh <= y <= by + (1 + pad) * bh):
            return True
    return False


def yolo_birdie_candidates(detections):
    out = []
    for d in detections:
        if d["label"].lower() in BIRDIE_LABELS:
            x, y, w, h = d["box"]
            out.append((x + w / 2.0, y + h / 2.0, d["conf"] * 255, False))
    return out


def load_tracknet_csv(path):
    #this is inlcluding the TrackNet-style CSV -> {frame: (x, y)} for visible frames
    pts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v for k, v in row.items()}
            if int(float(row.get("visibility", 1))) == 1:
                pts[int(row["frame"])] = (float(row["x"]), float(row["y"]), 255.0, False)
    return pts


class BirdieTracker:
    # uses the constant-velocity Kalman filter that picks the candidate closest to the
    #prediction and bridges short gaps (birdie hidden by a racket or a player)

    def __init__(self, fps, frame_shape, max_gap_s=0.15, confirm_hits=3):
        self.fps = fps
        h, w = frame_shape[:2]
        self.base_gate = 0.06 * max(h, w)# pixels
        # a birdie flying in front of a player moves fast; a slow "birdie" next to a
        # player is almost always the player's own motion (feet, arms, racket)
        self.min_body_speed = 0.6 * max(h, w) / fps # px per frame
        self.max_gap = max(2, int(max_gap_s * fps))
        self.confirm_hits = confirm_hits
        self.kf = None
        self.missed = 0
        self.hits = 0
        self.last_pt = None

    def _new_kf(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        kf.processNoiseCov[2:, 2:] *= 10
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        return kf

    def update(self, candidates):
        # candidates: [(x, y, score, in_body)]. Returns (x, y, status) where status is "seen", "predicted" or None
        if self.kf is None:
            free = [c for c in candidates if not c[3]]   # never start on a player's body
            if free:
                # start from the brightest candidate
                x, y = max(free, key=lambda c: c[2])[:2]
                self.kf = self._new_kf(x, y)
                self.hits, self.missed = 1, 0
            return None

        pred = self.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])
        gate = self.base_gate * (1 + self.missed) * (2.0 if self.hits < self.confirm_hits else 1.0)

        best, best_d = None, gate
        for x, y, _, in_body in candidates:
            d = np.hypot(x - px, y - py)
            if in_body:
                speed = float(np.hypot(pred[2, 0], pred[3, 0]))
                if (self.hits < self.confirm_hits or d > 0.5 * self.base_gate
                        or speed < self.min_body_speed):
                    continue
            if d < best_d:
                best, best_d = (x, y), d

        if best is not None:
            self.kf.correct(np.array([[best[0]], [best[1]]], np.float32))
            self.hits += 1
            self.missed = 0
            if self.hits >= self.confirm_hits:
                return (best[0], best[1], "seen")
            return None

        self.missed += 1
        if self.missed > self.max_gap or self.hits < self.confirm_hits:
            self.kf = None                           # lost -> re-acquire next frame
            self.hits = 0
            if candidates:
                return self.update(candidates)
            return None
        return (px, py, "predicted")
