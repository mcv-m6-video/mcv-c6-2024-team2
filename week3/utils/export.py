import cv2
import numpy as np
import os


def draw_flow(img, im2W, u, v, img1_name, img2_name, output):
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    hsv = np.zeros(img.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    img_name_out_1 = os.path.join(output, f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}_flow.jpg")
    img_name_out_2 = os.path.join(output, f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}_2Warped.jpg")

    cv2.imwrite(img_name_out_1, rgb)
    cv2.imwrite(img_name_out_2, im2W[:, :, ::-1] * 255)
