"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

import gc
import time
import PIL

import logging

log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import re

import torch
import torchvision

from ISR.models import RDN

import pytesseract
import imutils
from scipy.spatial import distance as dist

from skimage.restoration import estimate_sigma

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def _clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
rdn = RDN(weights='psnr-small')

def _cleanText(t):
    t = re.sub(r" +", " ", t) # replace multiple spaces with one
    t = re.sub(r"[\s\n]+\n", "\n", t).strip() # remove empty lines
    return t

def _biggest_contour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 58000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.015 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest

def _sharpnessTenengrad(img):
    # Calculate the gradient of the image using the Sobel operator
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)

    # Calculate the variance of the gradient image
    return grad.var()


def _estimateNoise(img):
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)

def _order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = np.sum(pts,axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[3] = pts[np.argmin(diff)]
    rect[1] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def _orderPointsClockwise(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl])


def _getAngle(img):
    image = imutils.resize(img, height = 500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 50, 150)


    cnts = None
    cnts, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

    screenCnt = None
    screenCnt = _biggest_contour(cnts)

    if(len(screenCnt)==0 ):
        edged = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,13,0.5)
        cnts = None
        cnts, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

        screenCnt = None
        screenCnt = _biggest_contour(cnts)

    if(len(screenCnt) >0 ):
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
        screenCnt = _order_points(screenCnt[:,0,:])
        angle = cv2.minAreaRect(screenCnt)[-1]
        angle = 90 - angle if (angle>45) else angle

        return angle
    else:
        return None

# pipeline de melhoramento
def _binarizeAdaptive(image, T=19, C=25, show=False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, T, C)
    return binarized

def _sharp(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(np.array(image), -1, kernel)

def _dilate(image):
        return cv2.dilate(np.array(image), np.zeros((5, 5), 'uint8'), iterations=4)

def _medianBlur(image, filter_size=3):
    return cv2.medianBlur(np.array(image), filter_size)

def _ISR(image):
    lr_img = np.array(image)
    sr_img = lr_img
    try:
        sr_img = rdn.predict(lr_img)
    except:
        pass
    return sr_img

# filtragem
def _filterImages(imagesList):

  imagesGood = []
  imagesMedium = []
  imagesBad = []

  for img in imagesList:

    imgBlurLevel = _sharpnessTenengrad(img)
    noiseLevel = _estimateNoise(img)
    imgAngle = _getAngle(img)

    #boas - aceitaveis
    if(imgBlurLevel >= 12111.474131839535 and noiseLevel<= 0.4516075407697839): #verifica ruido e nitidez
        if(imgAngle != None): # verifica se conseguiu calcular um angulo
            if(imgAngle <= 1.249346097310384):# se encaixa em todos os parametros classifica como bom
                imagesGood = np.append(imagesGood,i)
        else:
            imagesGood = np.append(imagesGood,i)


  for img in imagesList:
    if(img in imagesGood): continue

    imgBlurLevel = _sharpnessTenengrad(img)
    noiseLevel = _estimateNoise(img)
    imgAngle = _getAngle(img)

    #descartadas 
    if(imgAngle != None):
        if(noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798 or imgAngle >= 8.065917314340671):
            imagesBad = np.append(imagesBad,i)
        else:
            imagesMedium = np.append(imagesMedium,i)

    else:
        if((noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798)):
            imagesBad = np.append(imagesBad,i)
        else:
            imagesMedium = np.append(imagesMedium,i)

  return (imagesGood, imagesMedium, imagesBad)

def _imageQuality(image):
  img = np.array(image)
  imgBlurLevel = _sharpnessTenengrad(img)
  noiseLevel = _estimateNoise(img)
  imgAngle = _getAngle(img)
  #boas - aceitaveis
  if(imgBlurLevel >= 13099.15020419604 and noiseLevel<= 0.5727857190277632 ): #verifica ruido e nitidez
    if(imgAngle != None): # verifica se conseguiu calcular um angulo
      if(imgAngle <= 0.8458721126828875):# se encaixa em todos os parametros classifica como bom
        return "good"
    else:
      return "good"
  #descartadas
  if(imgAngle != None):
    if(noiseLevel >= 4.306616843268918 or imgBlurLevel <= 4444.8946011304815 or imgAngle >= 11.691148059708732 ):
      return "bad"
    else:
      return "medium"
  else:
    if((noiseLevel >= 4.306616843268918 or imgBlurLevel <= 4444.8946011304815 )):
      return "bad"
    else:
      return "medium"


#def _imageQuality(img):
#    imgBlurLevel = _sharpnessTenengrad(img)
#    noiseLevel = _estimateNoise(img)
#    imgAngle = _getAngle(img)
#    #boas - aceitaveis
#    if(imgBlurLevel >= 12111.474131839535 and noiseLevel<= 0.4516075407697839): #verifica ruido e nitidez
#        if(imgAngle != None): # verifica se conseguiu calcular um angulo
#            if(imgAngle <= 1.249346097310384):# se encaixa em todos os parametros classifica como bom
#                return "good"
#        else:
#            return "good"
#    #descartadas 
#    if(imgAngle != None):
#        if(noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798 or imgAngle >= 8.065917314340671):
#            return "bad"
#        else:
#            return "medium"
#    else:
#        if((noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798)):
#            return "bad"
#        else:
#            return "medium"

def _fileterImagePipeline(image):
    img = np.array(image)
    img_quality = _imageQuality(img)
    if img_quality == "good":
        return pytesseract.image_to_string(img)
    else:
        return ""

def _unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M

def _findDestinationPoints(src, shape):
    w_list = [
        abs(src[0][0] - src[1][0]),
        abs(src[2][0] - src[3][0])
    ]
    h_list = [
        abs(src[0][1] - src[2][1]),
        abs(src[1][1] - src[3][1])
    ]
    max_w = max(w_list) # maior lado horizontal do recibo
    max_h = max(h_list) # maior lado vertical do recibo

    h, w = shape[0], shape[1]
    inc_w = (w - max_w) / 2
    inc_h = (h - max_h) / 2

    return np.float32([(inc_w, inc_h),
                    (inc_w + max_w, inc_h),
                    (inc_w, inc_h + max_h),
                    (inc_w + max_w, inc_h + max_h)])

def _applyHomography(image, points):
    src = np.array(points, dtype="float32")
    dst = _findDestinationPoints(src, image.shape)   
    warped_img, M = _unwarp(image, src, dst)
    return warped_img, dst

def _generateMasks(image):
    # setup SAM
    input_point = np.array([[image.shape[1] / 2, image.shape[0] / 2]])
    input_label = np.array([1])
    predictor.set_image(image)
    # generate masks
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks, scores

def _segmentMaskHull(mask):
    gray = mask.astype(np.uint8) * 255
    edged = cv2.Canny(gray, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
    chosen_contours = cnt[0]
    for i in range(1, min(len(cnt), 4)):
        chosen_contours = np.concatenate([chosen_contours, cnt[i]], axis=0)
    hull = np.intp(cv2.convexHull(chosen_contours))
    epsilon = 0.1 * cv2.arcLength(hull, True)
    points = cv2.approxPolyDP(hull, epsilon, True)
    if len(points) != 4:
        return []
    points = _orderPointsClockwise(points.reshape(-1).reshape((-1, 2)))
    return points

def _findCorners(masks, scores):
    corners = []
    chosen_index = -1
    image_area = masks[0].shape[0] * masks[0].shape[1]
    for i in range(3):
        m = masks[i]
        s = scores[i]
        points = _segmentMaskHull(m)
        if len(points) != 4:
            continue
        contour_area = cv2.contourArea(points)
        if contour_area / image_area < 0.35:
            continue
        if len(corners) > 0:
            if chosen_index != -1:
                if s > scores[chosen_index]:
                    corners = points
                    chosen_index = i
            else:
                corners = points
                chosen_index = i
        else:
            corners = points
            chosen_index = i
    if len(corners) == 0:
        corners = _orderPointsClockwise(np.array([[0, 0], [masks[0].shape[0], 0], [masks[0].shape[0], masks[0].shape[1]], [0, masks[0].shape[1]]]))
    return corners

def _homography(img):
    masks, scores = _generateMasks(img)
    pts = _findCorners(masks, scores)
    return _applyHomography(img, np.array([pts[0], pts[1], pts[3], pts[2]]))[0]

def _adjustAllPipeline(image):
    img = np.array(image)
    img = _homography(img)
    img = _ISR(img)
    img = _medianBlur(img, filter_size=3)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def _adjustAllPipelineImg(image):
    img = np.array(image)
    img = _homography(img)
    img = _ISR(img)
    img = _medianBlur(img, filter_size=3)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return img

def _oursPipeline(image):
    img = np.array(image)
    img_quality = _imageQuality(img)
    if img_quality == "good":
        return pytesseract.image_to_string(img)
    elif img_quality == "bad":
        return ""
    else:
        return _adjustAllPipeline(img)

def apply_ours_pipeline(images):
    imgs = dict()
    for filename, loader in images.items():
        imgs[filename] = PIL.Image.fromarray(_adjustAllPipelineImg(loader()))
    return imgs

def generate_methods_ocr_text(images):
    all_results = dict()
    pipeline_functions = {
        "naive": lambda x: pytesseract.image_to_string(x),
        "only_filtering": _fileterImagePipeline,
        "adjust_all": _adjustAllPipeline,
        "ours": _oursPipeline,
    }
    for pipeline_name, pipeline_func in pipeline_functions.items():
        log.info("Running pipeline: %s" % pipeline_name)
        results = list()
        for file_name, loader in images.items():
            image = np.array(loader())
            _clean_gpu_memory()
            start = time.process_time()
            ocr_text = pipeline_func(image)
            pipeline_time = time.process_time() - start
            results.append({
                "file_name": file_name,
                "pipeline_time": pipeline_time,
                "ocr_text": ocr_text,
            })
        all_results[pipeline_name] = results
    return all_results

def ablation_sh(image):
    img = np.array(image)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def ablation_mp(image):
    img = np.array(image)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def ablation_sm(image):
    img = np.array(image)
    img = _medianBlur(img, filter_size=3)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def ablation_sr(image):
    img = np.array(image)
    img = _ISR(img)
    img = _medianBlur(img, filter_size=3)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def ablation_hm(image):
    img = np.array(image)
    img = _homography(img)
    img = _ISR(img)
    img = _medianBlur(img, filter_size=3)
    img = _dilate(img)
    img = _sharp(img)
    img = _binarizeAdaptive(img)
    return pytesseract.image_to_string(img)

def generate_ablation_ocr_text(images):
    all_results = dict()
    pipeline_functions = {
        "original": lambda x: pytesseract.image_to_string(x),
        "sh": ablation_sh,
        "mp": ablation_mp,
        "sm": ablation_sm,
        "sr": ablation_sr,
        "hm": ablation_hm,
    }
    for pipeline_name, pipeline_func in pipeline_functions.items():
        log.info("Running pipeline: %s" % pipeline_name)
        results = list()
        for file_name, loader in images.items():
            image = np.array(loader())
            _clean_gpu_memory()
            start = time.process_time()
            ocr_text = pipeline_func(image)
            pipeline_time = time.process_time() - start
            results.append({
                "file_name": file_name,
                "pipeline_time": pipeline_time,
                "ocr_text": ocr_text,
            })
        all_results[pipeline_name] = results
    return all_results

def generate_pipeline_sample_images(images):
    samples = dict()
    for filename, loader in images.items():
        img = np.array(loader())
        samples[filename + "_00_original.png"] = PIL.Image.fromarray(img)
        samples[filename + "_01_homography.png"] = PIL.Image.fromarray(_homography(img))
        samples[filename + "_02_isr.png"] = PIL.Image.fromarray(_ISR(samples[filename + "_01_homography.png"]))
        samples[filename + "_03_smooth.png"] = PIL.Image.fromarray(_medianBlur(samples[filename + "_02_isr.png"]))
        samples[filename + "_04_morphologic.png"] = PIL.Image.fromarray(_dilate(samples[filename + "_03_smooth.png"]))
        samples[filename + "_05_sharp.png"] = PIL.Image.fromarray(_sharp(samples[filename + "_04_morphologic.png"]))
    return samples

def generate_report(ocr_text, labels):
    for pipeline_name, ocr_results in ocr_text.items():
        for ocr_result in ocr_results:
            label = labels[ocr_result["file_name"]]()
