# -*- coding: utf-8 -*-
# Author: fkwong
# File: resize.py
# Date: 11/19/20

import cv2


def resize(img):
    h, w = img.shape[:2]
    scale = min(1, 720 / h, 1280 / w)
    return (
        cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        if scale != 1
        else img
    )
