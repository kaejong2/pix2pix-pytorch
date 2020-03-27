import numpy as np
import cv2
import os
import os.path



a = os.listdir('fundus')
print(a[0])

def img_load():
    for i in range(len(a)):
        img = cv2.imread('fundus/'+ a[i],cv2.IMREAD_COLOR)
        img_resize = cv2.resize(img,(565,584))
        # cv2.imshow('original',img_resize)
        # cv2.imshow('gray_img',gray2)
        # cv2.imshow('thr',thr)
        cv2.imwrite('binary/'+a[i],img_resize)
        print('binary/'+a[i])

img_load()
