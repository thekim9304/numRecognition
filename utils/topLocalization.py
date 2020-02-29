from tensorflow.keras.models import load_model
import numpy as np
import cv2

import time

def top_localization(img, models):
    o_h, o_w, o_c = img.shape[:]

    h_center = int(o_h / 2)
    t_region_img = img[:h_center,:]

    if o_c == 3:
        b8_img = cv2.cvtColor(t_region_img, cv2.COLOR_BGR2GRAY)
        b32_img = cv2.cvtColor(b8_img, cv2.COLOR_GRAY2BGR)
    else:
        b8_img = t_region_img.copy()
        b32_img = cv2.cvtColor(b8_img, cv2.COLOR_GRAY2BGR)

    # 절반으로 잘린 이미지에서의 좌표
    cand_box = get_cha_region_candbox(b8_img)
    if cand_box:
        cha_region = get_cha_region_box(b32_img, cand_box, models)
        if cha_region:
            n_coordi = get_num_region_box(b8_img, cha_region)
        else:
            n_coordi = []
    else:
        n_coordi = []

    return n_coordi

# input : gray scale img
# output : box axis list
def get_cha_region_candbox(img):
    ret, thr = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thr, (5, 5), 0)
    erosion = cv2.erode(blur, np.ones((1, 31), np.uint8), iterations=1)
    edge = cv2.Canny(erosion, 50, 150)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge)
    cand_box = []
    for label in range(nlabels):
        if label < 1:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if width >= 70 and width <= 150 and height >= 15 and height <= 50:
            cand_box.append((x, y, x+width, y+height))

    if cand_box:
        cand_box = checkIntersectAndCombine(cand_box)

        for i in range(len(cand_box)):
            cand_box[i] = list(cand_box[i])

    return cand_box

# input : 1차원 이미지
# output : 숫자 영역 좌표
def get_num_region_box(img, cha_region):
    img = img[cha_region[1]:cha_region[3], cha_region[0]:cha_region[2]]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, thr = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)

    for i in range(thr.shape[1]):
        s = sum(thr[:,i]) / thr.shape[0]

        if s != 255.0:
            save_s = i
            break

    for i in range(thr.shape[1]):
        s = sum(thr[:,thr.shape[1]-1-i]) / thr.shape[0]


        if s != 255.0:
            save_e = i
            break

    cha_region[0] = cha_region[0] + save_s
    cha_region[2] = cha_region[2] - save_e

    center = int((cha_region[2] - cha_region[0]) / 2) + cha_region[0]
    l_center = int(int((cha_region[2] - cha_region[0]) / 2) / 2) + cha_region[0] - 5
    r_center = int((cha_region[2] - cha_region[0]) / 2) + int(int((cha_region[2] - cha_region[0]) / 2) / 2) + cha_region[0]

    n_coordi = []
    n_coordi.append((cha_region[0], cha_region[1], l_center, cha_region[3]))
    n_coordi.append((l_center, cha_region[1], center, cha_region[3]))
    n_coordi.append((center, cha_region[1], r_center, cha_region[3]))
    n_coordi.append((r_center, cha_region[1], cha_region[2], cha_region[3]))

    return n_coordi

# input : 3채널 이미지 (흑백)
# output : final box
def get_cha_region_box(img, cand_box, model):
    cha_region_box = []
    if len(cand_box) >= 2:
        for i in range(len(cand_box)):
            check_region = img[cand_box[i][1]:cand_box[i][3], cand_box[i][0]:cand_box[i][2]]
            check_region = cv2.resize(check_region, (50, 100), interpolation=cv2.INTER_CUBIC)
            check_region = np.expand_dims(check_region, 0)

            yhat = model.predict(check_region)
            yhat = np.argmax(yhat)

            if yhat == 1:
                cha_region_box = cand_box[i]
    elif len(cand_box) == 1:
        cha_region_box = cand_box[0]

    return cha_region_box




################################################################################
import itertools

def get_coordi_in_stats(stats, label):
    area = stats[label, cv2.CC_STAT_AREA]
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    return x, y, width, height, area

# my Rectangle = (x1, y1, x2, y2), a bit different from OP's x, y, w, h
def intersection(rectA, rectB): # check if rect A & B intersect
    a, b = rectA, rectB
    startX = max( min(a[0], a[2]), min(b[0], b[2]) )
    startY = max( min(a[1], a[3]), min(b[1], b[3]) )
    endX = min( max(a[0], a[2]), max(b[0], b[2]) )
    endY = min( max(a[1], a[3]), max(b[1], b[3]) )
    if startX < endX and startY < endY:
        return True
    else:
        return False

def combineRect(rectA, rectB): # create bounding box for rect A & B
    a, b = rectA, rectB
    startX = min( a[0], b[0] )
    startY = min( a[1], b[1] )
    endX = max( a[2], b[2] )
    endY = max( a[3], b[3] )
    return (startX, startY, endX, endY)

def checkIntersectAndCombine(rects):
    if rects is None:
        return None
    mainRects = rects
    noIntersect = False
    while noIntersect == False and len(mainRects) > 1:
        mainRects = list(set(mainRects))
        # get the unique list of rect, or the noIntersect will be
        # always true if there are same rect in mainRects
        newRectsArray = []
        for rectA, rectB in itertools.combinations(mainRects, 2):
            newRect = []
            if intersection(rectA, rectB):
                newRect = combineRect(rectA, rectB)
                newRectsArray.append(newRect)
                noIntersect = False
                # delete the used rect from mainRects
                if rectA in mainRects:
                    mainRects.remove(rectA)
                if rectB in mainRects:
                    mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            # if no newRect is created = no rect in mainRect intersect
            noIntersect = True
        else:
            # loop again the combined rect and those remaining rect in mainRects
            mainRects = mainRects + newRectsArray
    return mainRects
