"""
This is a boilerplate pipeline 'image_perturbation'
generated using Kedro 0.18.7
"""

import re
import time

import logging

log = logging.getLogger(__name__)

from skimage.transform import rotate
from skimage.util import random_noise
from torchmetrics.functional import char_error_rate, word_error_rate
import cv2
import numpy as np
import pandas as pd
import pytesseract

def _cleanText(t):
    t = re.sub(r" +", " ", t) # replace multiple spaces with one
    t = re.sub(r"[\s\n]+\n", "\n", t).strip() # remove empty lines
    return t

def _generate_results(express_expense, labels, preprocess_image_func):
    results = list()
    for file_name, loader in express_expense.items():
        image = loader()
        image = preprocess_image_func(image)
        label = labels[file_name.replace(".jpg", ".txt")]()
        start = time.process_time()
        ocr_text = pytesseract.image_to_string(image)
        ocr_time = time.process_time() - start
        result = {"image_file": file_name,
                  "wer": float(word_error_rate(preds=_cleanText(ocr_text), target=_cleanText(label))),
                  "cer": float(char_error_rate(preds=_cleanText(ocr_text), target=_cleanText(label))),
                  "br": 1 if ocr_text else 0,
                  "ba": int(ocr_text == label),
                  "run_time": ocr_time}
        results.append(result)
        df_results = pd.DataFrame(results)
        df_results['ba'] = df_results['cer'].le(0.05).astype(int)
    return df_results

_perturbation_functions = {
    "original": lambda image: image,

    "rotate5": lambda image: (rotate(np.array(image), 5, resize=True) * 255).astype(np.uint8),
    "rotate10": lambda image: (rotate(np.array(image), 10, resize=True) * 255).astype(np.uint8),
    "rotate15": lambda image: (rotate(np.array(image), 15, resize=True) * 255).astype(np.uint8),
    "rotate20": lambda image: (rotate(np.array(image), 20, resize=True) * 255).astype(np.uint8),
    "rotate_m5": lambda image: (rotate(np.array(image), -5, resize=True) * 255).astype(np.uint8),
    "rotate_m10": lambda image: (rotate(np.array(image), -10, resize=True) * 255).astype(np.uint8),
    "rotate_m15": lambda image: (rotate(np.array(image), -15, resize=True) * 255).astype(np.uint8),
    "rotate_m20": lambda image: (rotate(np.array(image), -20, resize=True) * 255).astype(np.uint8),

    "bright10": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=10),
    "bright30": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=30),
    "bright50": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=50),
    "bright_m10": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=-10),
    "bright_m30": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=-30),
    "bright_m50": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1, beta=-50),

    "contrast25": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1.25, beta=0),
    "contrast50": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1.50, beta=0),
    "contrast75": lambda image: cv2.convertScaleAbs(np.array(image), alpha=1.75, beta=0),
    "contrast_m75": lambda image: cv2.convertScaleAbs(np.array(image), alpha=0.75, beta=0),
    "contrast_m50": lambda image: cv2.convertScaleAbs(np.array(image), alpha=0.50, beta=0),
    "contrast_m25": lambda image: cv2.convertScaleAbs(np.array(image), alpha=0.25, beta=0),

    "random_noise05": lambda image: (random_noise(np.array(image), mode="gaussian", var=0.005) * 255).astype(np.uint8),
    "random_noise10": lambda image: (random_noise(np.array(image), mode="gaussian", var=0.01) * 255).astype(np.uint8),
    "random_noise15": lambda image: (random_noise(np.array(image), mode="gaussian", var=0.015) * 255).astype(np.uint8),

    "pyr_scale1": lambda image: cv2.pyrDown(np.array(image)),
    "pyr_scale2": lambda image: cv2.pyrDown(cv2.pyrDown(np.array(image))),

    "gaussian_blur3": lambda image: cv2.GaussianBlur(np.array(image), (3, 3), 0),
    "gaussian_blur5": lambda image: cv2.GaussianBlur(np.array(image), (5, 5), 0),
    "gaussian_blur7": lambda image: cv2.GaussianBlur(np.array(image), (7, 7), 0)
}

def generate_ocr_perturbation_results(express_expense, labels):
    final_results = dict()
    for func_name, func in _perturbation_functions.items():
        log.info("Running perturbation: " + func_name)
        final_results[func_name] = _generate_results(express_expense, labels, func)
    return final_results

def generate_ocr_perturbation_report(image_perturbation_metrics):
    results = dict()
    for filename, loader in image_perturbation_metrics.items():
        df = loader().describe().loc[["mean", "std"], :].apply(lambda x: round(x * 100, 1))
        df["run_time"] = (df["run_time"] * 10).apply(round)
        df = df.T
        perturbation = filename.replace(".csv", "")
        results[perturbation] = df["mean"].astype(str) + " (" + df["std"].astype(str) + ")"
    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "perturbation"})#.sort_values("perturbation")
