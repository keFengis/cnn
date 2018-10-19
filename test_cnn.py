# -*- coding: utf-8 -*-
#!/usr/bin/python3
import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json


def perProcess(img):
    img = cv2.resize(img, (28,  28))
    img = (img.reshape(1, 28, 28 , 1)).astype('float32') / 255
    return img

path = '/home/kk/code/SVM/build/getImg/IMG_38_5.png'


model = model_from_json(open('model_digit.json').read())  
model.load_weights('model_digit.h5')

# image = cv2.imread(path, 0)
# img = cv2.imread(path, 0)
# img = cv2.resize(img, (28,  28))
# img = (img.reshape(1, 28, 28 , 1)).astype('float32') / 255

# predict = model.predict_classes(img)

# print ('识别为：')
# print (predict)

# cv2.imshow("Image1", image)
# cv2.waitKey(0)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    binary = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.inRange(binary, 0, 100)
    image, contours, hierarchy = cv2.findContours(dst,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for con in contours:
        x, y, w, h = cv2.boundingRect(con)
        if h<20:
            continue
        x = int(x - (33 - (w / 2)))
        y = int(y - (33 - (h / 2)));
        w = 65
        h = 65
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 0], 1)
        sample = dst[y : y + h, x : x + w]
        cv2.imshow('d', sample)
        sample = perProcess(sample)
        predict = model.predict_classes(sample)
        cv2.putText(frame, str(predict[0]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        cv2.imshow('frame', frame)
        cv2.imshow('binary', dst)
    cv2.waitKey(1)