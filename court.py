
#Court geometry + homography.

#Every sport is described in METERS with the origin at the FAR-LEFT corner (as seen from the camera): 
#x runs across the court (0 .. width), y runs toward the camera (0 = far baseline, length = near baseline).
#The net is at y = length / 2.

#Adding a new sport = adding one entry to COURTS. Nothing else changes,
#which is what makes the coordinate system "dynamic" (badminton / tennis /
#pickleball share the same pipeline).

import json
import os

import cv2
import numpy as np


def _rect(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def _badminton():
    L, W = 13.40, 6.10
    net = L / 2
    lines = _rect(0, 0, W, L)
    lines += [((0.46, 0), (0.46, L)), ((W - 0.46, 0), (W - 0.46, L))] # singles sidelines
    lines += [((0, 0.76), (W, 0.76)), ((0, L - 0.76), (W, L - 0.76))]   # doubles long service
    lines += [((0, net - 1.98), (W, net - 1.98)), ((0, net + 1.98), (W, net + 1.98))] # short service
    lines += [((W / 2, 0), (W / 2, net - 1.98)), ((W / 2, net + 1.98), (W / 2, L))] # centre lines
    lines += [((0, net), (W, net))]
    return dict(length=L, width=W, lines=lines,
                singles_inset=0.46,    # singles sideline offset
                singles_back_inset=0.0,  # singles rally uses full length
                doubles_back_inset=0.0) # doubles rally uses full length


def _tennis():
    L, W = 23.77, 10.97
    net = L / 2
    lines = _rect(0, 0, W, L)
    lines += [((1.37, 0), (1.37, L)), ((W - 1.37, 0), (W - 1.37, L))]
    lines += [((1.37, net - 6.40), (W - 1.37, net - 6.40)),
              ((1.37, net + 6.40), (W - 1.37, net + 6.40))]
    lines += [((W / 2, net - 6.40), (W / 2, net + 6.40))]
    lines += [((0, net), (W, net))]
    return dict(length=L, width=W, lines=lines, singles_inset=1.37,
                singles_back_inset=0.0, doubles_back_inset=0.0)


def _pickleball():
    L, W = 13.41, 6.10
    net = L / 2
    lines = _rect(0, 0, W, L)
    lines += [((0, net - 2.13), (W, net - 2.13)), ((0, net + 2.13), (W, net + 2.13))]  # kitchen
    lines += [((W / 2, 0), (W / 2, net - 2.13)), ((W / 2, net + 2.13), (W / 2, L))]
    lines += [((0, net), (W, net))]
    return dict(length=L, width=W, lines=lines, singles_inset=0.0,
                singles_back_inset=0.0, doubles_back_inset=0.0)


COURTS = {
    "badminton": _badminton(),
    "tennis": _tennis(),
    "pickleball": _pickleball(),
}

CORNER_ORDER = ["far-left", "far-right", "near-right", "near-left"]


class Court:
    #Maps betwn image pixels and court meters using the 4 outer corners

    def __init__(self, sport, image_corners):
        if sport not in COURTS:
            raise ValueError(f"Unknown sport '{sport}'. Options: {list(COURTS)}")
        self.sport = sport
        spec = COURTS[sport]
        self.length, self.width = spec["length"], spec["width"]
        self.lines = spec["lines"]
        self.singles_inset = spec["singles_inset"]
        self.net_y = self.length / 2

        self.image_corners = np.array(image_corners, dtype=np.float32).reshape(4, 2)
        court_corners = np.array([[0, 0], [self.width, 0],
                                  [self.width, self.length], [0, self.length]],
                                 dtype=np.float32)
        # image -> court (meters) and back
        self.H = cv2.getPerspectiveTransform(self.image_corners, court_corners)
        self.H_inv = np.linalg.inv(self.H)

        # pixel length of the court centre line, used to normalise arc heights
        far_mid = self.to_image([(self.width / 2, 0)])[0]
        near_mid = self.to_image([(self.width / 2, self.length)])[0]
        self.image_length_px = float(np.linalg.norm(near_mid - far_mid))

    # coordinate transforms 
    def to_court(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    #court logic 
    def inside(self, court_pt, margin=0.0, back_margin=None):
        """True if a court point lies inside the court (+ margin in meters)."""
        bm = margin if back_margin is None else back_margin
        x, y = court_pt
        return (-margin <= x <= self.width + margin) and (-bm <= y <= self.length + bm)

    def is_in(self, court_pt, game="doubles"):
        """In/out call for a landing point."""
        inset = self.singles_inset if game == "singles" else 0.0
        x, y = court_pt
        return inset <= x <= self.width - inset and 0 <= y <= self.length

    def side(self, court_pt):
        return "far" if court_pt[1] < self.net_y else "near"

    def dist_from_net(self, court_pt):
        return abs(court_pt[1] - self.net_y)

    def zone(self, court_pt):
        #Front/mid/rear x left/right from the player's own POV
        half = self.length / 2
        d = self.dist_from_net(court_pt)
        depth = "front" if d < half / 3 else ("mid" if d < 2 * half / 3 else "rear")
        x = court_pt[0]
        camera_left = x < self.width / 2
        # far-side players face the camera, so their left is camera-right
        player_left = camera_left if self.side(court_pt) == "near" else not camera_left
        return f"{self.side(court_pt)}-{depth}-{'left' if player_left else 'right'}"

    #drawing 
    def draw_on_frame(self, frame, color=(255, 255, 0), thickness=1):
        for a, b in self.lines:
            pa, pb = self.to_image([a, b])
            cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)), color, thickness, cv2.LINE_AA)
        return frame

    def map_canvas(self, px_per_m=40, margin_m=1.0):
        #Blank top-down court image + function converting court meters -> canvas px
        w = int((self.width + 2 * margin_m) * px_per_m)
        h = int((self.length + 2 * margin_m) * px_per_m)
        canvas = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)

        def to_px(pt):
            return (int((pt[0] + margin_m) * px_per_m), int((pt[1] + margin_m) * px_per_m))

        for a, b in self.lines:
            cv2.line(canvas, to_px(a), to_px(b), (255, 255, 255), 2, cv2.LINE_AA)
        return canvas, to_px


#calibration helpers
def corners_path_for(video_path):
    return os.path.splitext(video_path)[0] + "_corners.json"


def load_corners(path):
    with open(path) as f:
        data = json.load(f)
    return data["corners"]


def save_corners(path, corners, sport):
    with open(path, "w") as f:
        json.dump({"sport": sport, "order": CORNER_ORDER, "corners": corners}, f, indent=2)


def parse_corners_arg(text):
    #x1,y1 x2,y2 x3,y3 x4,y4' -> [[x1,y1], ...]
    pts = [[float(v) for v in p.split(",")] for p in text.split()]
    if len(pts) != 4:
        raise ValueError("--corners needs exactly 4 points: 'x,y x,y x,y x,y'")
    return pts


def calibrate_interactive(frame):
    #Click the 4 OUTER court corners in order: far-left, far-right, near-right, near-left.
    #r = reset, Enter/space = confirm, Esc = quit.
    pts = []
    win = "Click court corners (far-left, far-right, near-right, near-left)"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(view, CORNER_ORDER[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts)], len(pts) == 4, (0, 255, 255), 2)
        msg = (f"Click: {CORNER_ORDER[len(pts)]}" if len(pts) < 4
               else "Enter = confirm, r = reset")
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 32) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyWindow(win)
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(win)
    return pts
