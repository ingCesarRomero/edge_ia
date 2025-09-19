import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
image = cv2.imread("flia.jpg")
(h, w) = image.shape[:2]
start=time.time()
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
print(time.time()-start)
for i in range(detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > 0.15:
        print(f"rostro {confidence}")
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (x1,y1,x2,y2) = box.astype("int")
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite("result_dnn.jpg", image)