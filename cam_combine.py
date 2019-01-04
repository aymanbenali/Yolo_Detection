#!/usr/bin/env python
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import string
import random
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
import os
import datetime

os.putenv('DISPLAY', ':0.0')

options2 = {
    'model': 'cfg/tiny-yolo-voc.cfg',
    'load': 'bin/tiny-yolo-voc.weights',
    'threshold': 0.1,
}

tfnet2 = TFNet(options2)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

today = datetime.date.today()

smtpUser = 'abenali@myopla.com'
smtpPass = 'aymanbenali15'

toAdd = 'aymanbenali15@gmail.com'

fromAdd = smtpUser

subject  = 'Data File 01 %s' % today.strftime('%Y %b %d')
header = 'To :' + toAdd + '\n' + 'From : ' + fromAdd + '\n' + 'Subject : ' + subject + '\n'
body = 'This is a data file on %s' % today.strftime('%Y %b %d')

attach = 'Data on.csv'

def sendMail(to, subject, text, files=[]):
    assert type(to)==list
    assert type(files)==list

    msg = MIMEMultipart()
    msg['From'] = smtpUser
    msg['To'] = COMMASPACE.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach( MIMEText(text) )

    for file in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(file,"rb").read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"'
                       % os.path.basename(file))
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo_or_helo_if_needed()
    server.starttls()
    server.ehlo_or_helo_if_needed()
    server.login(smtpUser,smtpPass)
    server.sendmail(smtpUser, to, msg.as_string())

    print('Done')

    server.quit()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results2 = tfnet2.return_predict(frame)
        for color, result2 in zip(colors, results2):
            tl = (result2['topleft']['x'], result2['topleft']['y'])
            br = (result2['bottomright']['x'], result2['bottomright']['y'])
            label = result2['label']
            confidence2 = result2['confidence']
            if confidence2 > 0.7 and label == 'person' :   
                text = '{}: {:.0f}%'.format(label, confidence2 * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(
                    frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                mid_x = (result2['topleft']['x']+result2['topleft']['y'])/2
                mid_y = (result2['bottomright']['x']+result2['bottomright']['y'])/2
                apx_distance = round(((1 - (result2['topleft']['y'] - result2['topleft']['x']))**4),1)
                s, img = capture.read()
                char_set = string.ascii_uppercase + string.digits
                picName = ''.join(random.sample(char_set*6, 6))+'.jpg'
                cv2.imwrite('pics/'+picName, img)
                sendMail( [toAdd], subject, body, ['pics/'+picName] )
                #print(apx_distance)
                #print(tl,br)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
