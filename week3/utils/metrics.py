import numpy as np
import cv2

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width_intersection = max(0, x2 - x1)
    height_intersection = max(0, y2 - y1)

    area_intersection = width_intersection * height_intersection

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    area_union = area_box1 + area_box2 - area_intersection

    iou = area_intersection / area_union if area_union > 0 else 0.0
    return iou


def read_gt_flow(path):
    img_flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

    u = (img_flow[:, :, 2] - 2**15) / 64.0
    v = (img_flow[:, :, 1] - 2**15) / 64.0
    valid = img_flow[:, :, 0] == 1

    # Set invalid to 0
    u[valid == 0] = 0
    v[valid == 0] = 0

    return np.stack((u, v, valid), axis=2)


# Taken from Team 1 - 2021
def calculate_metrics(gt_flow:np.ndarray, pred_flow:np.ndarray, mask:np.ndarray=None, th:int=3):
    mask = gt_flow[:,:,2]  
    
    error = np.sqrt(np.sum((gt_flow[:,:,:2] - pred_flow[:,:,:2])**2, axis=-1))    
    msen = np.mean(error[mask != 0])
    pepn = 100 * np.sum(error[mask != 0] > th) / (mask != 0).sum()
    
    return msen, pepn