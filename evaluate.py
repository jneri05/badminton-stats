"""
Evaluation metrics from the proposal.

  Player detection : IoU, precision, recall, AP@0.5 (mAP for the single 'person' class)
  Birdie tracking  : precision, recall, F1 (TrackNet-style pixel tolerance)
  Landing position : error in meters vs your manual landing points
  Heatmap          : similarity between predicted and ground-truth heatmaps
  Speed            : error vs your hand-calculated speeds

Ground-truth CSV formats (make them from your annotations):
  players_gt.csv  : frame,x,y,w,h                 (pixel boxes, one row per player)
  birdie_gt.csv   : frame,visibility,x,y
  landing_gt.csv  : frame,x_m,y_m                 (frame where the birdie hit the floor)
  speed_gt.csv    : rally,shot,speed_mph          (hand-calculated)

Ex/
  python evaluate.py --results output/example_vid --players-gt players_gt.csv \
      --birdie-gt birdie_gt.csv --landing-gt landing_gt.csv --corners example_vid_corners.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def read_csv(path):
    with open(path) as f:
        return [{k.strip().lower(): v for k, v in r.items()} for r in csv.DictReader(f)]


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def player_metrics(detections, gt_rows, iou_thr=0.5):
    gt = defaultdict(list)
    for r in gt_rows:
        gt[int(r["frame"])].append([float(r[k]) for k in "xywh"])
    frames = set(gt)
    preds = []                                    # (conf, frame, box)
    for f in frames:
        if f < len(detections):
            for d in detections[f]["dets"]:
                preds.append((d["conf"], f, d["box"]))
    preds.sort(key=lambda p: -p[0])
    n_gt = sum(len(v) for v in gt.values())

    matched = {f: [False] * len(v) for f, v in gt.items()}
    tps, ious = [], []
    for conf, f, box in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt[f]):
            o = iou(box, g)
            if o > best and not matched[f][j]:
                best, best_j = o, j
        if best >= iou_thr:
            matched[f][best_j] = True
            tps.append(1)
            ious.append(best)
        else:
            tps.append(0)

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum([1 - t for t in tps])
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    # all-point interpolated AP (VOC / COCO style at one IoU threshold)
    mrec = np.concatenate([[0], recall, [1]])
    mpre = np.concatenate([[0], precision, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    tp = int(tp_cum[-1]) if len(tps) else 0
    return {"precision": round(tp / max(len(preds), 1), 4),
            "recall": round(tp / max(n_gt, 1), 4),
            "mean_iou_of_matches": round(float(np.mean(ious)), 4) if ious else 0.0,
            f"AP@{iou_thr}": round(ap, 4), "gt_boxes": n_gt, "pred_boxes": len(preds)}


def birdie_metrics(pred_rows, gt_rows, tol_px=10.0):
    pred = {int(r["frame"]): r for r in pred_rows}
    tp = fp = fn = tn = 0
    for g in gt_rows:
        f = int(g["frame"])
        gv = int(float(g["visibility"])) == 1
        p = pred.get(f)
        pv = p is not None and int(p["visibility"]) == 1
        if gv and pv:
            d = np.hypot(float(p["x"]) - float(g["x"]), float(p["y"]) - float(g["y"]))
            if d <= tol_px:
                tp += 1
            else:
                fp += 1
        elif pv and not gv:
            fp += 1
        elif gv and not pv:
            fn += 1
        else:
            tn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "tolerance_px": tol_px}


def landing_metrics(rallies, gt_rows, fps, max_dt_s=1.0):
    errs = []
    for g in gt_rows:
        gf = int(g["frame"])
        best = min(rallies, key=lambda r: abs(int(r["end_frame"]) - gf), default=None)
        if best is None or abs(int(best["end_frame"]) - gf) > max_dt_s * fps:
            continue
        errs.append(np.hypot(float(best["landing_x_m"]) - float(g["x_m"]),
                             float(best["landing_y_m"]) - float(g["y_m"])))
    return {"matched": len(errs), "of_gt": len(gt_rows),
            "mean_error_m": round(float(np.mean(errs)), 3) if errs else None,
            "median_error_m": round(float(np.median(errs)), 3) if errs else None}


def heatmap_metrics(pred_rows, gt_rows, corners_json, sport):
    """GT heatmap built from GT boxes' feet through the same homography."""
    from court import Court, load_corners
    from stats import heatmap_grid
    court = Court(sport, load_corners(corners_json))
    gt_feet = [(float(r["x"]) + float(r["w"]) / 2, float(r["y"]) + float(r["h"])) for r in gt_rows]
    gt_pts = court.to_court(gt_feet) if gt_feet else []
    frames = {int(r["frame"]) for r in gt_rows}
    pr_pts = [(float(r["court_x_m"]), float(r["court_y_m"])) for r in pred_rows
              if int(r["frame"]) in frames]
    a = heatmap_grid(pr_pts, court).ravel()
    b = heatmap_grid(gt_pts, court).ravel()
    if a.sum() == 0 or b.sum() == 0:
        return {"correlation": None}
    corr = float(np.corrcoef(a, b)[0, 1])
    l1 = float(np.abs(a / a.sum() - b / b.sum()).sum() / 2)   # 0 = identical, 1 = disjoint
    return {"correlation": round(corr, 4), "distribution_difference": round(l1, 4)}


def speed_metrics(shots, gt_rows):
    pred = {(int(s["rally"]), int(s["shot"])): float(s["avg_speed_mph"]) for s in shots}
    errs = [abs(pred[(int(g["rally"]), int(g["shot"]))] - float(g["speed_mph"]))
            for g in gt_rows if (int(g["rally"]), int(g["shot"])) in pred]
    return {"matched": len(errs), "mae_mph": round(float(np.mean(errs)), 2) if errs else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="output/<video> folder from main.py")
    p.add_argument("--players-gt")
    p.add_argument("--birdie-gt")
    p.add_argument("--landing-gt")
    p.add_argument("--speed-gt")
    p.add_argument("--corners", help="<video>_corners.json (needed for heatmap metric)")
    p.add_argument("--tol", type=float, default=10.0, help="birdie pixel tolerance")
    a = p.parse_args()

    with open(os.path.join(a.results, "stats.json")) as f:
        summary = json.load(f)
    report = {}
    if a.players_gt:
        with open(os.path.join(a.results, "detections.json")) as f:
            dets = json.load(f)
        gt = read_csv(a.players_gt)
        report["player_detection"] = player_metrics(dets, gt)
        if a.corners:
            report["heatmap"] = heatmap_metrics(read_csv(os.path.join(a.results, "player_boxes.csv")),
                                                gt, a.corners, summary["sport"])
    if a.birdie_gt:
        report["birdie"] = birdie_metrics(read_csv(os.path.join(a.results, "birdie.csv")),
                                          read_csv(a.birdie_gt), a.tol)
    if a.landing_gt:
        report["landing"] = landing_metrics(summary["rallies"], read_csv(a.landing_gt), summary["fps"])
    if a.speed_gt:
        report["speed"] = speed_metrics(read_csv(os.path.join(a.results, "shots.csv")),
                                        read_csv(a.speed_gt))
    print(json.dumps(report, indent=2))
    with open(os.path.join(a.results, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
