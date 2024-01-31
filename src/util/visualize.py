import cv2
import numpy as np


def flow2rgb(flow):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    hue = angle * 180 / np.pi / 2
    hue_normalized = cv2.normalize(hue, None, 0, 1, cv2.NORM_MINMAX)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue_normalized * 255
    hsv[..., 1] = magnitude_normalized * 255
    hsv[..., 2] = 255
    rgb = np.uint8(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    return rgb
