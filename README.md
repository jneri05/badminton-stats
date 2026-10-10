# Badminton Stats

## Introduction 
Analyze Badminton players to map their movements and shuttlecock landing position and speed. Applies a pretrained YOLO model to detect players on court. In progress: edge detection to recognize court lines, and training custom YOLO model to recognize shuttlecock in play.

## Output example
An output from a short badminton match clip
https://github.com/user-attachments/assets/6be8085c-af59-4032-9e91-ae1187127e71

## How to run on VS Code
1. Install requirements with  "pip install ultralytics opencv-python numpy"
2. Put your video in the same folder as `main.py`
3. In `main.py` change `"example_vid.MOV"` to match the name of your video
4. Run `main.py`
5. Initial run will make script export the model and creates `coco.names`, `yolov5su.onnx` in the folder.
6. The output video will be saved as 'output_example_vid.avi'. Change the name in line to fit your needs.
   
##  Known issues
- Object detection doesn't occur for all players on court especially during instances of occlusion, usually happens during doubles matches similar to output example above. 
- Additionally irrelevant players/spectators are detected
- Shuttlecock detection doesn't occur
- Output video format doesn't work for devices that don't support `.avi`
    
## Requirements
- Python 3
- `ultralytics`
- 'opencv'
- numpy
