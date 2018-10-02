#!/usr/bin/env python
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
os.putenv('DISPLAY', ':0.0')

options = {
    'model': 'cfg/tiny-yolo-voc-1c-last.cfg',
    'load': 2500,
    'threshold': 0.1,
}

options2 = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.1,
}

tfnet = TFNet(options)
tfnet2 = TFNet(options2)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 60)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)


while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        results2 = tfnet2.return_predict(frame)
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
            #print(apx_distance)
            #print(tl,br)
        for color, result2 in zip(colors, results2):
            tl = (result2['topleft']['x'], result2['topleft']['y'])
            br = (result2['bottomright']['x'], result2['bottomright']['y'])
            label = result2['label']
            confidence2 = result2['confidence']
            if confidence2 > 0.5 :   
                text = '{}: {:.0f}%'.format(label, confidence2 * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(
                    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                mid_x = (result2['topleft']['x']+result2['topleft']['y'])/2
                mid_y = (result2['bottomright']['x']+result2['bottomright']['y'])/2
                apx_distance = round(((1 - (result2['topleft']['y'] - result2['topleft']['x']))**4),1)
                #print(apx_distance)
                #print(tl,br)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
