from utils2.topLocalization import top_localization
from utils2.botLocalization import bot_localization

from tensorflow.keras.models import load_model
import numpy as np
import cv2

def num_recognition(img, models):
    t_coordi = top_localization(img, models[0])
    b_coordi = bot_localization(img, models[1])

    if t_coordi:
        t_n_string = extract_num_string(img, t_coordi, models[-1])
    else:
        t_n_string = 'None'

    if b_coordi:
        b_n_string = extract_num_string(img, b_coordi, models[-1])
    else:
        b_n_string = 'None'

    return t_n_string, b_n_string, t_coordi, b_coordi

def extract_num_string(img, coordi, model):
    if coordi:
        num_l = []
        for i in range(len(coordi)):
            iimg = img[coordi[i][1]:coordi[i][3], coordi[i][0]:coordi[i][2]]
            iimg = pre_img(iimg)

            yhat = model.predict(iimg)
            yhat = np.argmax(yhat)

            num_l.append(yhat)


        num_string = str(num_l[0]) + str(num_l[1]) + str(num_l[2]) + str(num_l[3])
    else:
        num_string = 'None'

    return num_string

def pre_img(img):
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    m_v = mean_v(img)

    if m_v <= 80:
        thr = 30
        if int(m_v) == 60 or int(m_v) == 59:
            thr = 55
    elif m_v > 80 and m_v <= 150:
        thr = 50
    elif m_v > 150 and m_v <= 190:
        thr = 120
    else:
        thr = 200

    ret, img = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32') / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.expand_dims(img, 0)

    return img

def mean_v(img):
    m = 0
    for i in range(img.shape[0]):
        r_m = np.mean(img[i])

        m = m + r_m

    m = m / img.shape[0]

    return m
