#Badminton Stats - main pipeline.

#Baseline kept from the first version: yolov5su.onnx exported with ultralytics on the
#first run, run through cv2.dnn, same thresholds (0.5 conf / 0.45 NMS), same box drawing.

#Added on top:
#  1. court mapping (homography from 4 corners)       -> court.py
#  2. player tracking + spectator filtering + heatmap  -> players.py, stats.py
 # 3. birdie tracking (classical / YOLO / TrackNet)     -> birdie.py
 # 4. rallies, landing positions, speeds, shot types   -> rally.py



import argparse
import os

import cv2
import numpy as np

from birdie import (BirdieTracker, MotionBirdieDetector, load_tracknet_csv,
                    yolo_birdie_candidates)
from court import (Court, calibrate_interactive, corners_path_for, load_corners,
                   parse_corners_arg, save_corners)
from players import HogDetector, PlayerTracker, YoloDetector
from rally import analyse
from stats import (player_color, player_stats, render_court_map, render_player_heatmap,
                   write_csv, write_json)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Badminton Stats")
    p.add_argument("--video", default="example_vid.MOV")
    p.add_argument("--out", default="output", help="output folder")
    p.add_argument("--sport", default="badminton", choices=["badminton", "tennis", "pickleball"])
    p.add_argument("--game", default="doubles", choices=["singles", "doubles"])
    p.add_argument("--corners", help="'x,y x,y x,y x,y' far-left, far-right, near-right, near-left")
    p.add_argument("--detector", default="yolo", choices=["yolo", "hog"])
    p.add_argument("--model", default="yolov5su.onnx")
    p.add_argument("--names", default="coco.names")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.45,
                   help="raise to ~0.6 to keep overlapping doubles players")
    p.add_argument("--hog-svm", help="custom HOG SVM from train_hog_svm.py")
    p.add_argument("--birdie", default="motion", choices=["motion", "yolo", "tracknet"])
    p.add_argument("--tracknet-csv", help="TrackNet output CSV (Frame,Visibility,X,Y)")
    p.add_argument("--codec", default="mp4v", help="mp4v (plays everywhere) or MJPG for .avi")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    p.add_argument("--reuse", action="store_true", help="reuse cached detections")
    p.add_argument("--max-frames", type=int, default=0, help="process only N frames (testing)")
    return p.parse_args(argv)


def get_court(args, first_frame):
    if args.corners:
        corners = parse_corners_arg(args.corners)
    elif os.path.exists(corners_path_for(args.video)):
        corners = load_corners(corners_path_for(args.video))
    else:
        print("Click the 4 outer court corners: far-left, far-right, near-right, near-left")
        corners = calibrate_interactive(first_frame)
    save_corners(corners_path_for(args.video), corners, args.sport)
    return Court(args.sport, corners)


#pass 1
def detect_pass(args, court, detector, fps, frame_shape):
    """Runs the heavy models once and returns raw per-frame results."""
    cap = cv2.VideoCapture(args.video)
    motion = MotionBirdieDetector(frame_shape) if args.birdie == "motion" else None
    tracknet = load_tracknet_csv(args.tracknet_csv) if args.birdie == "tracknet" else None
    btracker = BirdieTracker(fps, frame_shape)
    frames = []
    idx = 0
    while True:
        ret, image = cap.read()
        if not ret or (args.max_frames and idx >= args.max_frames):
            break
        dets = detector.detect(image)
        person_boxes = [d["box"] for d in dets if d["label"] in ("person", "player")]

        if args.birdie == "motion":
            cands = motion.detect(image, person_boxes)
        elif args.birdie == "yolo":
            cands = yolo_birdie_candidates(dets)
        else:
            cands = [tracknet[idx]] if idx in tracknet else []
        b = btracker.update(cands)

        frames.append({"dets": [d for d in dets if d["label"] in ("person", "player")],
                       "birdie": list(b) if b else None})
        idx += 1
        if idx % 100 == 0:
            print(f"  detected {idx} frames")
    cap.release()
    return frames


#analysis
def analyse_frames(args, court, raw, fps, frame_h=None):
    tracker = PlayerTracker(court, fps, max_players=4 if args.game == "doubles" else 2)
    players = {}
    for f, fr in enumerate(raw):
        tracks = tracker.update([dict(d) for d in fr["dets"]], f)
        players[f] = [{"id": t.id, "court": t.pos.tolist(), "box": t.box,
                       "coasted": t.missed > 0} for t in tracks]

    birdie = [tuple(fr["birdie"]) if fr["birdie"] else None for fr in raw]
    tracks = tracker.all_tracks()
    pstats = player_stats(tracks, court, fps)
    rallies, shots = analyse(birdie, players, court, fps, args.game,
                             classify=(args.sport == "badminton"), frame_h=frame_h)
    return players, birdie, tracks, pstats, rallies, shots


def per_frame_speeds(tracks, fps, window_s=0.25):
    w = max(1, int(window_s * fps))
    speeds = {}
    for pid, hist in tracks:
        for i in range(w, len(hist)):
            dt = (hist[i][0] - hist[i - w][0]) / fps
            if dt > 0:
                v = np.hypot(hist[i][1] - hist[i - w][1], hist[i][2] - hist[i - w][2]) / dt
                speeds[(pid, hist[i][0])] = v
    return speeds


# pass 2
def render_pass(args, court, players, birdie, tracks, rallies, shots, fps, out_path):
    cap = cv2.VideoCapture(args.video)
    writer = None
    speeds = per_frame_speeds(tracks, fps)
    shot_at = {}
    for s in shots:
        for f in range(s["start_frame"], s["end_frame"] + 1):
            shot_at[f] = s
    rally_at = {f: r for r in rallies for f in range(r["start_frame"], r["end_frame"] + 1)}
    trail = []
    mini, to_px = court.map_canvas(px_per_m=18, margin_m=0.5)
    landed = []

    f = 0
    while True:
        ret, image = cap.read()
        if not ret or f >= len(players):
            break
        height, width = image.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*args.codec),
                                     fps, (width, height))

        court.draw_on_frame(image)

        # players (same green-box style as the baseline, now with IDs + speed)
        m = mini.copy()
        for p in players[f]:
            if p["coasted"]:
                continue
            x, y, w, h = p["box"]
            col = player_color(p["id"])
            v = speeds.get((p["id"], f), 0.0)
            cv2.rectangle(image, (x, y), (x + w, y + h), col, 2)
            cv2.putText(image, f"P{p['id']} {v * 2.23694:.1f} mph", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.circle(m, to_px(p["court"]), 5, col, -1)

        # birdie trail
        b = birdie[f]
        if b and b[2] == "seen":
            trail.append((int(b[0]), int(b[1])))
        elif b is None:
            trail.clear()
        trail = trail[-int(fps * 0.5):]
        for i in range(1, len(trail)):
            cv2.line(image, trail[i - 1], trail[i], (0, 255, 255), 2)
        if b:
            cv2.circle(image, (int(b[0]), int(b[1])), 6, (0, 255, 255), 2)

        # landings revealed when each rally ends
        for r in rallies:
            if r["end_frame"] == f:
                landed.append(r)
        for r in landed:
            col = (0, 255, 0) if r["landing_in"] else (0, 0, 255)
            cv2.drawMarker(m, to_px((r["landing_x_m"], r["landing_y_m"])), col,
                           cv2.MARKER_TILTED_CROSS, 8, 2)

        # HUD
        hud = []
        if f in rally_at:
            hud.append(f"Rally {rally_at[f]['rally']}")
        if f in shot_at:
            s = shot_at[f]
            hud.append(f"P{s['hitter_id']}: {s['type']}  {s['avg_speed_mph']:.0f} mph")
        for i, t in enumerate(hud):
            cv2.putText(image, t, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(image, t, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # minimap in the top-right corner
        scale = min(1.0, 0.3 * height / m.shape[0])
        m = cv2.resize(m, None, fx=scale, fy=scale)
        mh, mw = m.shape[:2]
        if mh < height and mw < width:
            image[10:10 + mh, width - mw - 10:width - 10] = m

        writer.write(image)
        f += 1
    cap.release()
    if writer is not None:
        writer.release()


# main
def run(args, detector=None):
    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read video: {args.video}")

    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    court = get_court(args, first)

    cache = os.path.join(out_dir, "detections.json")
    if args.reuse and os.path.exists(cache):
        import json
        with open(cache) as fh:
            raw = json.load(fh)
        print(f"Reusing {len(raw)} cached frames")
    else:
        if detector is None:
            detector = (HogDetector(args.hog_svm) if args.detector == "hog" else
                        YoloDetector(args.model, args.names, conf_threshold=args.conf,
                                     nms_threshold=args.nms))
        print("Pass 1: detecting players and birdie ...")
        raw = detect_pass(args, court, detector, fps, first.shape)
        write_json(cache, raw)

    print("Analysing rallies, landings, speeds and shots ...")
    players, birdie, tracks, pstats, rallies, shots = analyse_frames(args, court, raw, fps, first.shape[0])

    # outputs
    cv2.imwrite(os.path.join(out_dir, "court_map.png"),
                render_court_map(court, tracks, rallies, title=f"{name}: heatmap + landings"))
    for pid, hist in tracks:
        cv2.imwrite(os.path.join(out_dir, f"heatmap_P{pid}.png"),
                    render_player_heatmap(court, pid, hist))
    write_csv(os.path.join(out_dir, "rallies.csv"), rallies)
    write_csv(os.path.join(out_dir, "shots.csv"), shots)
    write_csv(os.path.join(out_dir, "player_boxes.csv"),
              [{"frame": f, "id": p["id"], "x": p["box"][0], "y": p["box"][1],
                "w": p["box"][2], "h": p["box"][3], "court_x_m": round(p["court"][0], 2),
                "court_y_m": round(p["court"][1], 2)}
               for f, ps in players.items() for p in ps if not p["coasted"]])
    write_csv(os.path.join(out_dir, "birdie.csv"),
              [{"frame": f, "visibility": int(b is not None and b[2] == "seen"),
                "x": round(b[0], 1) if b else "", "y": round(b[1], 1) if b else ""}
               for f, b in enumerate(birdie)])
    summary = {
        "video": args.video, "sport": args.sport, "game": args.game, "fps": fps,
        "frames": len(raw), "players": pstats, "rallies": rallies,
        "shot_type_counts": {t: sum(s["type"] == t for s in shots)
                             for t in sorted({s["type"] for s in shots})},
    }
    write_json(os.path.join(out_dir, "stats.json"), summary)

    if not args.no_video:
        ext = ".avi" if args.codec.upper() == "MJPG" else ".mp4"
        out_path = os.path.join(out_dir, f"output_{name}{ext}")
        print("Pass 2: writing annotated video ...")
        render_pass(args, court, players, birdie, tracks, rallies, shots, fps, out_path)

    print(f"Done. {len(rallies)} rallies, {len(shots)} shots. Results in {out_dir}/")
    return summary


if __name__ == "__main__":
    run(parse_args())
