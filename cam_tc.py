#!/usr/bin/env python
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
from func import convert_hls,select_rgb_white_yellow,canny_edge,detect_edges,filter_region,select_region,hough_lines,draw_lines,average_slope_intercept,make_line_points,lane_lines,draw_lane_lines

os.putenv('DISPLAY', ':0.0')

options = {
    'model': 'cfg/tiny-yolo-voc-1c-last.cfg',
    'load': 2500,
    'threshold': 0.1,
}



tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


while True:
    try:
        stime = time.time()
        ret, frame = capture.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_yellow = select_rgb_white_yellow(frame)
        houg = hough_lines(select_region(detect_edges(white_yellow)))
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(
                    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                mid_x = (result['topleft']['x']+result['topleft']['y'])/2
                mid_y = (result['bottomright']['x']+result['bottomright']['y'])/2
                apx_distance = round(((1 - (result['topleft']['y'] - result['topleft']['x']))**4),1)
                if mid_x>130 and mid_y>460:
                    print("Danger from Right")
                elif mid_x<70 and mid_y>350:
                    print("Danger from Left")
                elif mid_x<90 and mid_y>420:
                    print("Danger from the midlle")
                else:
                    print("Safe")
        cv2.imshow('Detection_Only', frame)
        cv2.imshow('Detection_Combine_With_White_Line',draw_lane_lines(frame,lane_lines(frame,houg)))          
          #cv2.imshow('last',draw_lane_lines(frame,lane_lines(frame,houg)))
            #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    except(TypeError,OverflowError):
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
