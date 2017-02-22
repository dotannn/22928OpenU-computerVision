#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:38:26 2017

@author: dotan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image




##q2
#
#Fs = 8000
#f = 5
#sample = 8000
#x = np.arange(sample)
#y = np.sin(2 * np.pi * f * x / Fs)
#plt.figure()
#
#plt.plot(x, y)
#plt.xlabel('voltage(V)')
#plt.ylabel('sample(n)')
#plt.show()
#
#buf = io.BytesIO()
#plt.savefig(buf, format='png')
#buf.seek(0)
#im = Image.open(buf)
#
#pil_image = im.convert('RGB')
#
#open_cv_image = np.array(pil_image) 


img2 = np.zeros((500,500,3),'uint8')
img = np.zeros((500,500),'uint8')
px = int(5)
for i in range(30):
    px += int(round(400/ 30))
    cv2.line(img, (px,px), (px,px+50), color=255)
    

lines = cv2.HoughLines(img,1,np.pi/180,50)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

plt.figure()
plt.subplot(1,2,1), plt.imshow(img,'gray'), plt.title('edges image')
plt.subplot(1,2,2), plt.imshow(img2), plt.title('most dominate line')
plt.show()

