#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:41:32 2018

@author: trio_pu
"""

import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import cv2
import numpy as np


class ColorDetector(QDialog):
    def __init__(self):
        super(ColorDetector, self).__init__()
        loadUi('OpenCV.ui', self)
        self.image = None
        self.start_button.clicked.connect(self.start_webcam)
        self.stop_button.clicked.connect(self.stop_webcam)
        
        self.track_button.setCheckable(True)
        self.track_button.toggled.connect(self.track_webcam)
        self.track_enabled = False
        
        self.color1_button.clicked.connect(self.set_color1)
        self.color2_button.clicked.connect(self.set_color2)
        
    def track_webcam(self, status):
        if status:
            self.track_enabled = True
            self.track_button.setText('Stop Tracking')
        else:
            self.track_enabled = False
            self.track_button.setText('Track Color')
            
    def set_color1(self):
        self.color1_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()],np.uint8)
        self.color1_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()],np.uint8)
        
        self.color1_value.setText('C1 -> Min :'+str(self.color1_lower)+' Max: '+str(self.color1_upper))
    
    def set_color2(self):
        self.color2_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()],np.uint8)
        self.color2_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()],np.uint8)
        
        self.color2_value.setText('C2 -> Min :'+str(self.color2_lower)+' Max: '+str(self.color2_upper))
        
    
    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
       
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
        
    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image,1)
        self.displayImage(self.image,1)
        
        #Reference
        #lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
        #upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        color_lower = np.array([self.h_min.value(), self.s_min.value(), self.v_min.value()], np.uint8)
        color_upper = np.array([self.h_max.value(), self.s_max.value(), self.v_max.value()], np.uint8)
        
        self.current_value.setText('Current Value -> Min :'+str(color_lower)+' Max: '+str(color_upper))

        # print('Min :'+str(color_lower)+' Max: '+str(color_upper))
        color_mask = cv2.inRange(hsv, color_lower, color_upper)

        self.displayImage(color_mask,2)
        
        if(self.track_enabled and self.color_1.isChecked()):
            trackedImage = self.track_colored_object(self.image.copy())
            self.displayImage(trackedImage,1)
        else:
            self.displayImage(self.image,1)
        
    def track_colored_object(self,img):
        blur = cv2.blur(img,(3,3))
        hsv  = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        if(self.color_1.isChecked()):
            color_mask = cv2.inRange(hsv, self.color1_lower, self.color1_upper)
            
            erode = cv2.erode(color_mask, None, iterations=2)
            dilate = cv2.dilate(erode,None, iterations=10)
            
            kernelOpen = np.ones((5,5))
            kernelClose = np.ones((20,20))
            
            maskOpen = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernelOpen)
            maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
            
            (_,contours,hierarchy) = cv2.findContours(maskClose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if(area>5000):
                    x,y,w,h = cv2.boundingRect(cnt)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(img,'Color 1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
        
        if(self.color_2.isChecked()):
            color_mask = cv2.inRange(hsv, self.color2_lower, self.color2_upper)
            
            erode = cv2.erode(color_mask, None, iterations=2)
            dilate = cv2.dilate(erode,None, iterations=10)
            
            kernelOpen = np.ones((5,5))
            kernelClose = np.ones((20,20))
            
            maskOpen = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernelOpen)
            maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
            
            (_,contours,hierarchy) = cv2.findContours(maskClose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if(area>5000):
                    x,y,w,h = cv2.boundingRect(cnt)
                    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(img,'Color 2',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),1)
                         
        return img
        
    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()
        
    
    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3: #[0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0],qformat)
        #BGR to RGB
        outImage = outImage.rgbSwapped()
        
        if window == 1:
            self.image_label1.setPixmap(QPixmap.fromImage(outImage))
            self.image_label1.setScaledContents(True)
        
        if window ==2:
            self.image_label2.setPixmap(QPixmap.fromImage(outImage))
            self.image_label2.setScaledContents(True)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ColorDetector()
    window.setWindowTitle('OpenCV Color Detector')
    window.show()
    sys.exit(app.exec_())
