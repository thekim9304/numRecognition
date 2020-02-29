from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
import cv2

import time

def bot_localization(img, models):
    o_h, o_w, o_c = img.shape[:]

    h_center = int(o_h / 2)
    b_region_img = img[h_center:,:]

    if o_c == 3:
        b8_img = cv2.cvtColor(b_region_img, cv2.COLOR_BGR2GRAY)
        b32_img = cv2.cvtColor(b8_img, cv2.COLOR_GRAY2BGR)
    else:
        b8_img = t_region_img.copy()
        b32_img = cv2.cvtColor(b8_img, cv2.COLOR_GRAY2BGR)

    # 절반으로 잘린 이미지에서의 좌표
    n_coordi = []
    # start = time.time()
    cand_box = get_cha_region_candbox(b8_img, [10, 20], models)
    # print("b_1_time : {}".format(time.time() - start))
    if cand_box:
        # start = time.time()
        num_box = get_num_region_box(b32_img, cand_box)
        # print("b_2_time : {}".format(time.time() - start))
        if num_box:
            # start = time.time()
            n_coordi = get_num_box(num_box)
            # print("b_3_time : {}".format(time.time() - start))

            for i in range(len(n_coordi)):
                n_coordi[i] = list(n_coordi[i])

                n_coordi[i][1] = n_coordi[i][1] + h_center
                n_coordi[i][3] = n_coordi[i][3] + h_center
        else:
            n_coordi = []
    else:
        n_coordi = []

    return n_coordi

## Step 1
def get_cha_region_candbox(img, threshold, model):
    boxs = []
    for thr in threshold:
        region_cand = get_region_candbox(img, thr)
        for box in region_cand:
            in_img = img[box[1]:box[3], box[0]:box[2]]
            in_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR)
            in_img = cv2.resize(in_img, (100, 50), interpolation=cv2.INTER_CUBIC)
            in_img = np.expand_dims(in_img, 0)

            yhat = model.predict(in_img)
            yhat = np.argmax(yhat)

            if yhat == 1:
                boxs.append(box)

    if boxs:
        x1 = boxs[0][0]
        y1 = boxs[0][1]
        x2 = boxs[0][2]
        y2 = boxs[0][3]
        for box in boxs:
            if box[0] < x1:
                x1 = box[0]
            if box[1] < y1:
                y1 = box[1]
            if box[2] > x2:
                x2 = box[2]
            if box[3] > y2:
                y2 = box[3]

        return [x1, y1, x2, y2]
    else:
        return []

def get_region_candbox(img, threshold):
    kernel = np.ones((1, 17), np.uint8)
    x = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    ret, x = cv2.threshold(x, threshold, 255, cv2.THRESH_BINARY)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    x = cv2.dilate(x, np.ones((3, 13), np.uint8), iterations=1)
    x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.Canny(x, 50, 150)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(x)

    region_candbox = []
    for label in range(nlabels):
        if label < 1:
            continue

        x_, y_, width, height, area = get_coordi_in_stats(stats, label)

        if area > 50 and width <= 190 and height <= 70:
            region_candbox.append((x_, y_, x_+width, y_+height))

    return region_candbox

## Step 2
def get_num_region_box(img, n_box):
    y1 = n_box[1]
    y2 = n_box[3]
    x1 = n_box[0]
    x2 = n_box[2]

    n_region_img = img[y1:y2, x1:x2]

    if n_region_img.shape[2] == 3:
        n_region_img = cv2.cvtColor(n_region_img, cv2.COLOR_BGR2GRAY)

    x_s, x_e = get_binary_coordi(n_region_img, y1, y2, x1, x2)
    x_s_, x_e_ = get_num_region(img, y1, y2, x_s, x_e)

    x1_ = x_s_ + x_s
    y1_ = y1
    x2_ = x_e_ + x_s
    y2_ = y2

    num_box = [x1_, y1_, x2_, y2_]

    return num_box

def get_binary_coordi(img, y1, y2, x1, x2):
    img = img[+int(y1*0.1):, :]
    m_value = mean_v(img)

    if m_value < 150:
        if int(m_value) == 81:
            thr = 25
        else:
            hist, _ = np.histogram(img.flatten(), 256, [0, 256])

            for i in range(hist.shape[0]):
                if hist[i+1] - hist[i] < -5:
                    f_max = hist[i]
                    idx = i
                    break

            np.where(hist == max(hist[idx+1:-3]))
            s_max_idx = np.where(hist == max(hist[idx+1:-3]))[0][0]
            thrs = np.where(hist == min(hist[idx:s_max_idx]))
            for th in thrs[0]:
                if th > idx:
                    thr = th
                    break
    else:
        img = img[:-int(y2*0.1),:]
        img = preprocessing_bright(img)
        thr = 170

    ret, th = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

    for r in range(th.shape[1]):
        if np.mean(th[:, r]) != 255:
            x_s = r
            break
    for r in range(th.shape[1]):
        if np.mean(th[:, th.shape[1]-1-r]) != 255:
            x_e = th.shape[1]-1-r
            break

    x_s = x_s - 5 + x1
    x_e = x_e + 5 + x1

    return x_s, x_e

def get_num_region(img, y1, y2, x_s, x_e):
    x = img[y1:y2-int(y2*0.1), x_s:x_e]
    x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = cv2.erode(x, np.ones((1, 5), np.uint8), iterations=1)

    m_v = mean_v(x)

    if m_v < 45:
        thr = 25
    elif m_v < 80:
        thr = 30
    elif m_v >= 80 and  m_v < 140:
        thr = 70
    elif m_v >= 140 and m_v < 158:
        thr = 110
    else:
        thr = 160

    ret, th = cv2.threshold(x, thr, 255, cv2.THRESH_BINARY)

    if np.mean(th[0, :]) != 255 or np.mean(th[1, :]) != 255:
        concat = False
        del_idx = []
        for c in range(th.shape[0]):
            if c != 0 and np.mean(th[c, :]) == 255:
                concat = True
                break
            del_idx.append(c)
        if concat == True:
            for idx in del_idx:
                th[idx, :] = 255

    x_s_ = -1
    x_e_ = -1

    for r in range(th.shape[1]):
        if np.mean(th[:, r]) != 255:
            x_s_ = r
            break
    for r in range(th.shape[1]):
        if np.mean(th[:, th.shape[1]-1-r]) != 255:
            x_e_ = th.shape[1]-1-r
            break

    if (x_e_ - x_s_) < (x.shape[1] / 3):
        x_s_ = 0
        x_e_ = x.shape[1]

    if x_s_ == -1:
        x_s_ = 0
    if x_e_ == -1:
        x_e_ = x.shape[1]

    x_s_ = x_s_ - 5
    x_e_ = x_e_ + 5

    return x_s_, x_e_

## Step 3

def get_num_box(num_box):
    center = int((num_box[2] - num_box[0]) / 2) + num_box[0]
    l_center = int(int((num_box[2] - num_box[0]) / 2) / 2) + num_box[0]
    r_center = int((num_box[2] - num_box[0]) / 2) + int(int((num_box[2] - num_box[0]) / 2) / 2) + num_box[0]

    n_coordi = []
    n_coordi.append((num_box[0], num_box[1], l_center, num_box[3]))
    n_coordi.append((l_center, num_box[1], center, num_box[3]))
    n_coordi.append((center, num_box[1], r_center, num_box[3]))
    n_coordi.append((r_center, num_box[1], num_box[2], num_box[3]))

    return n_coordi



# input : gray scale
def preprocessing_bright(img):
    total_mean = 0
    for i in range(img.shape[0]):
        col_mean = int(sum(img[i]) / img.shape[1])

        total_mean = total_mean + col_mean

    total_mean = int(total_mean / img.shape[0])

    plus_value = 255 - total_mean

    min_v = 256
    max_v = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res_value = img[i][j] + plus_value

            if res_value > 250:
                res_value = 255

            img[i][j] = res_value

    return img

def mean_v(img):
    m = 0
    for i in range(img.shape[0]):
        r_m = np.mean(img[i])

        m = m + r_m

    m = m / img.shape[0]

    return m

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
