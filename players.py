
#Player detection + tracking.

#YoloDetector: the original baseline (yolov5su.onnx through cv2.dnn) moved into a
# class. The decoding math is unchanged from the first main.py.
#HogDetector: classical baseline from the proposal (HOG + SVM). Uses OpenCV's
#built-in people SVM, or your own SVM trained with train_hog_svm.py.
#PlayerTracker: keeps a stable ID per player, filters out spectators using the
# court homography, and coasts through short occlusions.

import os

import cv2
import numpy as np


class YoloDetector:
#same structure as the orginal main.py in 

    def __init__(self, onnx_path="yolov5su.onnx", names_path="coco.names",
                 input_size=640, conf_threshold=0.5, nms_threshold=0.45):
        if not os.path.exists(onnx_path) or not os.path.exists(names_path):
            # identical to the baseline's first-run export
            from ultralytics import YOLO
            model = YOLO("yolov5s.pt")
            if not os.path.exists(onnx_path):
                model.export(format="onnx")
            if not os.path.exists(names_path):      # don't overwrite an existing file
                with open(names_path, "w") as f:
                    for i in range(len(model.names)):
                        f.write(model.names[i] + "\n")

        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        with open(names_path) as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0,
                                     size=(self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        predictions = self.net.forward()[0].T

        boxes, confidences, class_ids = [], [], []
        x_factor = width / self.input_size
        y_factor = height / self.input_size
        for detection in predictions:
            class_scores = detection[4:]
            class_id = int(np.argmax(class_scores))
            confidence = class_scores[class_id]
            if confidence > self.conf_threshold:
                cx, cy, w, h = detection[:4]
                boxes.append([int((cx - w / 2) * x_factor), int((cy - h / 2) * y_factor),
                              int(w * x_factor), int(h * y_factor)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        out = []
        for i in np.array(indices).flatten():
            out.append({"box": boxes[i], "conf": confidences[i],
                        "label": self.classes[class_ids[i]]})
        return out


class HogDetector:
    #Classical HOG + SVM people detector (proposal baseline)."""

    def __init__(self, svm_path=None, scale=1.05, hit_threshold=0.0):
        self.hog = cv2.HOGDescriptor()
        if svm_path:
            coef = np.load(svm_path)          # produced by train_hog_svm.py
            self.hog.setSVMDetector(coef.astype(np.float32))
        else:
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.scale = scale
        self.hit_threshold = hit_threshold

    def detect(self, image, max_side=960):
        # HOG is slow on 4K phone footage, so run on a downscaled copy
        h, w = image.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = cv2.resize(image, None, fx=s, fy=s) if s < 1 else image
        rects, weights = self.hog.detectMultiScale(small, hitThreshold=self.hit_threshold,
                                                   winStride=(8, 8), padding=(8, 8),
                                                   scale=self.scale)
        out = []
        for (x, y, bw, bh), wgt in zip(rects, np.array(weights).flatten()):
            out.append({"box": [int(x / s), int(y / s), int(bw / s), int(bh / s)],
                        "conf": float(wgt), "label": "person"})
        return out


def foot_point(box):
    #Bottom-centre of a box = where the player stands on the ground plane.
    x, y, w, h = box
    return (x + w / 2.0, y + h)


class Track:
    def __init__(self, tid, court_pos, box, frame_idx):
        self.id = tid
        self.pos = np.array(court_pos, dtype=float)# smoothed court meters
        self.vel = np.zeros(2)
        self.box = box
        self.missed = 0
        self.history = [(frame_idx, float(self.pos[0]), float(self.pos[1]), False)]


class PlayerTracker:
    #Greedy nearest-neighbour tracker in COURT coordinates.

    #Working in meters (not pixels) means the gating distance is the same for a
    #player at the far baseline and one right in front of the camera.
    

    def __init__(self, court, fps, max_players=4, court_margin=1.0, back_margin=2.0,
                 max_missed_s=2.0, smooth=0.5):
        self.court = court
        self.fps = fps
        self.max_players = max_players
        self.court_margin = court_margin
        self.back_margin = back_margin
        self.max_missed = int(max_missed_s * fps)
        self.smooth = smooth
        self.tracks = []
        self.finished = []
        self.next_id = 1

    def all_tracks(self):
        #Every track ever created, with trailing coasted points removed
        out = []
        for t in self.finished + self.tracks:
            hist = list(t.history)
            while hist and hist[-1][3]:
                hist.pop()
            if len(hist) >= max(5, int(0.5 * self.fps)):   # ignore flicker tracks
                out.append((t.id, hist))
        return out

    def filter_on_court(self, detections):
        #Drop spectators / people on other courts (Known issue #2)."""
        kept = []
        for d in detections:
            if d["label"] not in ("person", "player"):
                continue
            cp = self.court.to_court([foot_point(d["box"])])[0]
            if self.court.inside(cp, self.court_margin, self.back_margin):
                d = dict(d, court=cp)
                kept.append(d)
        kept.sort(key=lambda d: -d["conf"])
        return kept[: self.max_players + 2]

    def update(self, detections, frame_idx):
        dets = self.filter_on_court(detections)

        # predict (constant velocity, damped while coasting through occlusion)
        for t in self.tracks:
            t.pred = t.pos + t.vel * (0.9 ** t.missed)

        # greedy matching by court distance; gate grows while a track is missing
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = 1.0 + 0.05 * t.missed
            for di, d in enumerate(dets):
                dist = np.linalg.norm(t.pred - d["court"])
                same_side = self.court.side(t.pred) == self.court.side(d["court"])
                if dist < gate and (same_side or dist < 0.5):   # players don't cross the net
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            new_pos = self.smooth * t.pos + (1 - self.smooth) * np.asarray(d["court"])
            t.vel = new_pos - t.pos
            t.pos = new_pos
            t.box = d["box"]
            t.missed = 0
            t.history.append((frame_idx, float(t.pos[0]), float(t.pos[1]), False))

        # unmatched tracks coast on their prediction (occlusion handling)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
                t.pos = t.pred
                t.history.append((frame_idx, float(t.pos[0]), float(t.pos[1]), True))
        self.finished += [t for t in self.tracks if t.missed > self.max_missed]
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        # new tracks only while we have room for more players
        for di, d in enumerate(dets):
            if di in used_d or len(self.tracks) >= self.max_players:
                continue
            if all(np.linalg.norm(t.pos - d["court"]) > 0.8 for t in self.tracks):
                self.tracks.append(Track(self.next_id, d["court"], d["box"], frame_idx))
                self.next_id += 1
        return self.tracks
