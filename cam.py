#!/usr/bin/env python
import cv2
import freenect
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



def video_cv(video):
    return video[:, :, ::-1]

def get_video():
    return video_cv(freenect.sync_get_video()[0])

    
tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(cv2.CAP_OPENNI)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)


while True:
    stime = time.time()
    ret, frame = capture.read()
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

        cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
