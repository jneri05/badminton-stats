import os
import cv2
import numpy as np


if not os.path.exists("yolov5su.onnx") or not os.path.exists("coco.names"):
    from ultralytics import YOLO

    model = YOLO("yolov5s.pt")
    model.export(format="onnx")

    with open("coco.names", "w") as f:
        for i in range(len(model.names)):
            f.write(model.names[i] + "\n")

net = cv2.dnn.readNetFromONNX("yolov5su.onnx")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

input_size = 640
confidence_threshold = 0.5
nms_threshold = 0.45

cap = cv2.VideoCapture("example_vid.MOV")

fps = cap.get(cv2.CAP_PROP_FPS)
writer = None

while True:
    ret, image = cap.read()
    if not ret:
        break

    height, width = image.shape[:2]

    if writer is None:
        writer = cv2.VideoWriter(
            "output_example_vid.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height)
        )

    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1/255.0, size=(input_size, input_size),
        swapRB=True, crop=False
    )

    net.setInput(blob)
    predictions = net.forward()

    predictions = predictions[0]
    predictions = predictions.T

    boxes, confidences, class_ids = [], [], []
    x_factor = width / input_size
    y_factor = height / input_size

    for detection in predictions:
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence > confidence_threshold:
            cx, cy, w, h = detection[:4]
            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            box_width = int(w * x_factor)
            box_height = int(h * y_factor)
            boxes.append([left, top, box_width, box_height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences,
        score_threshold=confidence_threshold,
        nms_threshold=nms_threshold
    )

    for i in indices:
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        confidence = confidences[i]
        label = f"{classes[class_id]}: {confidence:.2f}"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    writer.write(image)

cap.release()
writer.release()