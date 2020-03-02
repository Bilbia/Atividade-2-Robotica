#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
from ipywidgets import widgets, interact, interactive, FloatSlider, IntSlider
from matplotlib import pyplot as plt
import numpy as np
import time as t
import sys
import math
import auxiliar as aux
import imutils

cap = cv2.VideoCapture(0)

img_color = cv2.imread("folha_atividade.png")
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)



magenta = "#EA148D"
cyan = "#01AFEC"


cor1, cor2 = aux.ranges(cyan)
mag1 = np.array([ 170, 50, 50], dtype=np.uint8)
mag2 = np.array([ 200, 255, 255], dtype=np.uint8)
cy1 = np.array([ 100, 50, 50], dtype=np.uint8)
cy2 = np.array([ 110, 255, 255], dtype=np.uint8)


print(cor1,cor2)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist

font = cv2.FONT_HERSHEY_SIMPLEX



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    

    #detectar cores
    mascara_1 = cv2.inRange(hsv, mag1, mag2)
    mascara_2 = cv2.inRange(hsv, cy1, cy2)
    masc_mg = cv2.GaussianBlur(mascara_1, (5,5),0)
    masc_cy = cv2.GaussianBlur(mascara_2, (5,5),0)
    mask = masc_mg + masc_cy


 



    #Circulos Hough
    blur = cv2.GaussianBlur(gray,(5,5),0)
    bordas = auto_canny(blur)
    circles = []
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=80,minRadius=5,maxRadius=60)

    KNOWN_DISTANCE = 40.0
    KNOWN_WIDTH = 14.0
    # marker = cv2.minAreaRect(c)
    # focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print(focalLength)

    pontos = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
            pontos.append((i[0],i[1]))
            if len(pontos) > 1:
                cv2.line(frame,(pontos[0]),(pontos[1]),(255,0,0),5)
                x1 = int(pontos[1][0])
                y1 = int(pontos[1][1])
                x2 = int(pontos[0][0])
                y2 = int(pontos[0][1])
                dist_line = calculateDistance(x1, y1,x2,y2)
                foc = (175 * KNOWN_DISTANCE)/KNOWN_WIDTH
                dist_real = (KNOWN_WIDTH * foc)/dist_line
                tg1 = -1
                if x1 != x2:
                    tg2 = (y1 - y2)/(x1 - x2)
                angulo = math.degrees(np.arctan(tg1) - np.arctan(tg2)) + 90
                cv2.putText(frame,'Angle : {}'.format(angulo),(0,450), font, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,'Distance : {} cm'.format(dist_real),(0,400), font, 1,(0,255,255),2,cv2.LINE_AA)            
                pontos = []


    contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"


    #distancia 
    # cnts = cv2.findContours(bordas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # c = max(cnts, key = cv2.contourArea)

    # def distance_to_camera(knownWidth, focalLength, perWidth):
	#     # compute and return the distance from the maker to the camera
	#     return (knownWidth * focalLength) / perWidth
    




    # Display the resulting frame q 1
   # np.concatenate(frame,mask,bordas_color)
    cv2.imshow('colorido hihihi', frame)
    cv2.imshow('MASK', mask )
    cv2.imshow('Bordas', bordas_color)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

