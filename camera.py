# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 11:30:54 2021

@author: arunkothari
"""

import numpy as np
import cv2


canvas = None
x1, y1 = 0, 0


cap = cv2.VideoCapture(0)
color_circle = (0, 255, 0)

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Main rectangle
    m_start_point = (447, 19) 
    m_end_point = (622, 218) 
    color = (255, 0, 0)
    thickness = 2
    
    # Hand thres window
    frame_temp = np.copy(frame[m_start_point[1]:253,
                       m_start_point[0]:m_end_point[0]])
    
    cv2.rectangle(frame, m_start_point, m_end_point, color, thickness) 

    
    # Green rectangle
    start_point = (587, 218) 
    end_point = (622, 253) 
    color = (0, 255, 0)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness) 
    
    # Yellow rectangle
    start_point = (552, 218) 
    end_point = (587, 253) 
    color = (0, 255, 255)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness) 
    
    # White rectangle
    start_point = (517, 218) 
    end_point = (552, 253) 
    color = (255, 255, 255)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness) 
    
    # Red rectangle
    start_point = (482, 218) 
    end_point = (517, 253) 
    color = (0, 0, 255)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness) 

    # Cyan rectangle
    start_point = (447, 218) 
    end_point = (482, 253) 
    color = (255, 255, 0)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness)
    
    # Converting skin to HSV
    hsvim = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 60], dtype = "uint8")
    upper = np.array([20, 150, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2,2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding the max contour(HAND)
    max_area = 0
    if len(contours) > 0:
        for i in range(len(contours)):
            cont=contours[i]
            area=cv2.contourArea(cont)
            if(area>max_area):
                max_area = area
                largest_contour=i
    try:
        c = contours[largest_contour]

    except IndexError:
        continue
        
    except NameError:
        continue
        
    
    extTop = tuple(c[c[:, :, 1].argmin()][0])

    
    if 218<extTop[1]<253:
        if  0<extTop[0]<35:
            color_circle = (255, 255, 0)
        if  35<extTop[0]<70:
            color_circle = (0, 0, 255)
        if  70<extTop[0]<105:
            color_circle = (255, 255, 255)
        if  105<extTop[0]<140:
            color_circle = (0, 255, 255)
        if  140<extTop[0]<175:
            color_circle = (0, 255, 0)
    
    cv2.circle(thresh, extTop, 8, color_circle, -1)
    
    if x1 == 0 and y1 == 0:
        x1,y1= extTop
    
    else:
    # Draw the line on the canvas
        canvas = cv2.line(canvas, (x1,y1),extTop, color_circle, 4)
        
    x1, y1 = extTop
    
    cv2.imshow("Canvas", canvas)

    
    frame = cv2.drawContours(frame, c, 0, (0,255,255), 3)
    cv2.imshow("thresh", thresh)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

