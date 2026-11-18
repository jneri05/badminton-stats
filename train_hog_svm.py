#Train the classical HOG + SVM player detector (proposal: classical baseline).

#positives/ : cropped images of players (any size, resized to 64x128)
#negatives/ : frames or crops with NO players (court, walls, spectators' seats ...)

#Step 1 trains a linear SVM, step 2 runs it on the negatives and retrains with the
#false positives it finds (hard-negative mining), as described in the proposal.

#python train_hog_svm.py --pos data/positives --neg data/negatives --out hog_svm.npy
#python main.py --detector hog --hog-svm hog_svm.npy

import argparse
import glob
import os
import random

import cv2
import numpy as np

WIN = (64, 128)


def images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    return sorted(p for e in exts for p in glob.glob(os.path.join(folder, e)))


def random_windows(img, n):
    h, w = img.shape[:2]
    if h < WIN[1] or w < WIN[0]:
        img = cv2.resize(img, (max(w, WIN[0]), max(h, WIN[1])))
        h, w = img.shape[:2]
    for _ in range(n):
        x, y = random.randint(0, w - WIN[0]), random.randint(0, h - WIN[1])
        yield img[y:y + WIN[1], x:x + WIN[0]]


def train(features, labels, C=0.01):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(C)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
    svm.train(np.array(features, np.float32), cv2.ml.ROW_SAMPLE, np.array(labels, np.int32))
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    return np.append(sv[0], -rho).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pos", required=True)
    p.add_argument("--neg", required=True)
    p.add_argument("--out", default="hog_svm.npy")
    p.add_argument("--neg-per-image", type=int, default=10)
    a = p.parse_args()

    hog = cv2.HOGDescriptor()
    feats, labels = [], []
    for path in images(a.pos):
        img = cv2.resize(cv2.imread(path), WIN)
        feats.append(hog.compute(img).ravel())
        labels.append(1)
        feats.append(hog.compute(cv2.flip(img, 1)).ravel())     # augmentation
        labels.append(1)
    negs = [cv2.imread(p) for p in images(a.neg)]
    for img in negs:
        for win in random_windows(img, a.neg_per_image):
            feats.append(hog.compute(win).ravel())
            labels.append(-1)
    print(f"Round 1: {labels.count(1)} positive, {labels.count(-1)} negative windows")
    detector = train(feats, labels)

    # hard-negative mining: every detection on a negative image is a false positive
    hog.setSVMDetector(detector)
    hard = 0
    for img in negs:
        rects, _ = hog.detectMultiScale(img, winStride=(8, 8))
        for x, y, w, h in rects:
            crop = cv2.resize(img[y:y + h, x:x + w], WIN)
            feats.append(hog.compute(crop).ravel())
            labels.append(-1)
            hard += 1
    print(f"Round 2: added {hard} hard negatives")
    if hard:
        detector = train(feats, labels)
    np.save(a.out, detector)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
