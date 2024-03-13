from pyflow import pyflow
import os
import cv2
import numpy as np
import time
from PIL import Image
import re
from utils import read_flow, calculate_metrics
import optuna


INPUT = './seq45'
GT = INPUT + '_gt'

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def process_img(path, filename, resize=False):
    img_path = os.path.join(path, filename)
    img = Image.open(img_path)
    img = np.array(img)
    img = img.astype(float) / 255.0

    return img


def objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 0.1)
    ratio = trial.suggest_float('ratio', 0.1, 0.9)
    minWidth = trial.suggest_int('minWidth', 10, 50)
    nOuterFPIterations = trial.suggest_int('nOuterFPIterations', 1, 10)
    nInnerFPIterations = trial.suggest_int('nInnerFPIterations', 1, 5)
    nSORIterations = trial.suggest_int('nSORIterations', 20, 40)

    msen, pepn = run_pyflow(alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations)

    return msen, pepn


def run_pyflow(alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType = 0):
    imgs = sorted(os.listdir(INPUT), key=natural_sort_key)
    gts = sorted(os.listdir(GT), key=natural_sort_key)
    for img1_name, img2_name, gt_name in zip(imgs[:-1], imgs[1:], gts):
        img1 = process_img(INPUT, img1_name)
        img2 = process_img(INPUT, img2_name)

        s = time.time()
        u, v, im2W = pyflow.coarse2fine_flow(
            img1, img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType)
        e = time.time()

        print('Time Taken for calculating optical flow: %.2f seconds for image of size (%d, %d, %d)' % (
            e - s, img1.shape[0], img1.shape[1], img1.shape[2]))

        gt_flow = read_flow(os.path.join(GT, gt_name))
        pred_flow = np.dstack((u, v))
        msen, pepn = calculate_metrics(gt_flow, pred_flow)
        print(f"Pyflow metrics, MSEN:{msen}, PEPN:{pepn}")

        return msen, pepn

study = optuna.create_study(directions=['minimize', 'minimize'])
study.optimize(objective, n_trials=50)

