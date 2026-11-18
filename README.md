# Badminton Stats

## Introduction
Analyze badminton players from a single phone video to map their movements, the shuttlecock's landing positions and speeds, and the type of each shot. A pretrained YOLO model detects players on court (the original baseline), a court homography turns pixels into meters, and a birdie tracker finds rallies, hits and landings. The same pipeline also works for tennis and pickleball courts.

## Output example
An output video from a short badminton match clip showing player detection

https://github.com/user-attachments/assets/6be8085c-af59-4032-9e91-ae1187127e71

## Features

Featurs + what you get + output

Player detection (baseline) | YOLOv5su via `cv2.dnn`, same thresholds as v1 | `players.py` |
Court mapping | 4 clicked corners → homography (pixels ↔ meters) | `court.py` |Spectator filtering | Only people standing on/near the court are kept | `players.py` |
Player tracking | Stable IDs (P1, P2 …), coasts through short occlusions | `players.py` |
Player heatmap | Where each player spends the most time + zone % (front/mid/rear, left/right) | `stats.py` |
Player speed | Avg / max speed (mph) and distance covered | `stats.py` |
Birdie tracking | Classical: frame differencing + Kalman filter. Deep: fine-tuned YOLO class or TrackNet CSV | `birdie.py` |
| Birdie landing positions | Landing point (meters) of every rally + in/out call | `rally.py` |
| Shot speed | Average speed of every shot (mph) | `rally.py` |
| Shot types (stretch) | smash, clear, drop, net shot, lift, drive | `rally.py` |
| Other sports | `--sport tennis` / `--sport pickleball` | `court.py` |
| Evaluation | IoU, precision, recall, AP, F1, landing / heatmap / speed error | `evaluate.py` |

## How to run on VS Code
1. Install requirements with `pip install -r requirements.txt`
2. Put your video in the same folder as `main.py`
3. Run `python main.py --video your_video.MOV` (with no arguments it still uses `example_vid.MOV`)
4. Initial run will export the model and create `coco.names`, `yolov5su.onnx` in the folder (same as before).
5. A window opens on the first frame: click the 4 **outer** court corners in this order: **far-left, far-right, near-right, near-left**, then press Enter. They are saved to `<video>_corners.json` so you only click once per video.
6. Results are saved in `output/<video name>/`.

Useful options:

python main.py --video match.MOV --game singles          # 2 players, singles in/out lines
python main.py --video match.MOV --sport pickleball      # tennis | pickleball | badminton
python main.py --video match.MOV --nms 0.6               # keeps overlapping doubles players
python main.py --video match.MOV --reuse                 # skip detection, redo analysis only
python main.py --video match.MOV --corners "512,300 1410,302 1800,1010 120,1005"   # no clicking (Colab)
python main.py --video match.MOV --detector hog          # classical player detector
python main.py --video match.MOV --birdie tracknet --tracknet-csv match_ball.csv
python main.py --video match.MOV --codec MJPG            # old .avi output


## Output files (`output/<video>/`)
| File | Contents |
|---|---|
| `output_<video>.mp4` | Annotated video: court lines, player IDs + speed, birdie trail, shot type, mini court map |
| `court_map.png` | 2D court with player heatmap and landing dots labelled with speed (mph); green = in, red = out |
| `heatmap_P<n>.png` | Heatmap for each player |
| `stats.json` | Per-player stats (distance, speed, most-used zone), rallies, shot-type counts |
| `rallies.csv` | One row per rally: landing x/y (m), in/out, final shot, who won the point |
| `shots.csv` | One row per shot: hitter, start/end (m), duration, speed (mph), arc, type |
| `player_boxes.csv`, `birdie.csv` | Per-frame tracks (used by `evaluate.py`) |
| `detections.json` | Cached detections so `--reuse` skips the slow model |

Court coordinates are meters with the origin at the far-left corner; x goes across the court, y goes toward the camera, and the net is at y = 6.7 m (badminton).

## Classical vs deep (proposal comparison)
- **Players:** `--detector yolo` (deep) vs `--detector hog` (HOG + SVM). Train your own SVM with hard-negative mining:
  `python train_hog_svm.py --pos data/positives --neg data/negatives --out hog_svm.npy`, then `--detector hog --hog-svm hog_svm.npy`.
- **Birdie:** `--birdie motion` (frame differencing + Kalman) vs `--birdie yolo` (a fine-tuned YOLO with a `shuttlecock` class) vs `--birdie tracknet` (TrackNet's CSV output: `Frame,Visibility,X,Y`). All three feed the same tracker, so the comparison is fair.
- **Fine-tuned YOLO:** train with ultralytics on your annotated frames (classes `player`, `shuttlecock`), export to ONNX, then run `--model best.onnx --names my_classes.names --birdie yolo`.

## Evaluation
Make ground-truth CSVs from your annotations, then:
```
python evaluate.py --results output/example_vid \
    --players-gt players_gt.csv --birdie-gt birdie_gt.csv \
    --landing-gt landing_gt.csv --speed-gt speed_gt.csv --corners example_vid_corners.json
```
| GT file | Columns |
|---|---|
| `players_gt.csv` | `frame,x,y,w,h` |
| `birdie_gt.csv` | `frame,visibility,x,y` |
| `landing_gt.csv` | `frame,x_m,y_m` |
| `speed_gt.csv` | `rally,shot,speed_mph` (hand-calculated) |

## Known issues
- Detection still misses players during heavy occlusion in doubles; the tracker coasts for up to 2 s, and `--nms 0.6` helps keep overlapping players.
- Shuttle speed is the **average horizontal speed** of a shot (distance between hitter and receiver/landing ÷ flight time). A single camera can't measure height, so this is lower than the peak speed a radar would show for smashes.
- The landing point is the last place the birdie was seen. If it disappears behind a player just before landing, the point is a little off (`landing_confidence` = `low` when the birdie wasn't falling).
- Shot-type thresholds (`SHOT_RULES` in `rally.py`) are heuristics; tune them on your validation videos. Shot types are badminton-only.
- The classical birdie detector needs a fixed camera (tripod), and fast camera shake causes false positives.

## Requirements
- Python 3
- `ultralytics`
- `opencv-python`
- `numpy`
