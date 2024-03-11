from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("../yolo-weights/yolov8n.pt")

while True:
    x, img = cap.read()
    res = model(img, stream = True)
    for r in res:
        boxes = r.boxes
        for box in boxes:

            # Here we are finding Bounding Boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1, y1), (x2, y2), (255,0,255),3) # 3 is thichness here ans (255,0,255) is color of box

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1,y1,w,h))

            # Here we are finding confidence
            conf = (math.ceil(box.conf[0]) * 100) / 100  # * 100 and /100 is done to show upto 2 decimal ppints

            # Here we are finding class of object
            cls = box.cls[0]

            # Here we are putting confidence and class of object on box of object
            cvzone.putTextRect(img, f'{cls}{conf}', (max(0,x1), max(35, y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
