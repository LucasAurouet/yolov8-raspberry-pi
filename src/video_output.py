import cv2
from ultralytics import YOLO
import torch
from picamera2 import Picamera2
import time

def inference(model, frame):

    results = model(frame)    

    return results[0].boxes

def draw_boxes(frame, boxes):
        for box, conf, label in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            # Draw the box
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the image 
        cv2.imshow('video', frame)

def read_stream(model=None):

    picam2 = Picamera2()

    # Configuration "still" ou "preview"
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)

    picam2.start()
    time.sleep(2)
    
    while True:
        frame = cv2.resize(picam2.capture_array(), 640, 640)
        
        if model != None:
            boxes = inference(model, frame)
            draw_boxes(frame, boxes)
        else:    
            cv2.imshow('video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break  
    
    picam2.stop()
    cv2.destroyAllWindows()
    
model = YOLO('C:\\Users\\Lucas\\Desktop\\yolov8-realtime-detection\\weights\\finetuned_weights.pt')

img_test = cv2.imread("/home/lucas/Desktop/yolov8-raspberry-pi/test_image.jpg")
results = model(img_test)
draw_boxes(img_test, results[0].boxes)

# read_stream(model=model)