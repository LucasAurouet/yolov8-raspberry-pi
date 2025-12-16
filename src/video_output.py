import cv2
from ultralytics import YOLO
import torch

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

    stream = cv2.VideoCapture(0)

    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    
    while True:
        ret, frame = stream.read()
        fps = stream.get(cv2.CAP_PROP_FPS)
        
        if not ret:
            print('invalid stream')
            break
        
        if model != None:
            boxes = inference(model, frame)
            draw_boxes(frame, boxes)
        else:    
            cv2.imshow('video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break  
    stream.release()
    cv2.destroyAllWindows()
    print(f'{fps} FPS')
    
model = YOLO('C:\\Users\\Lucas\\Desktop\\yolov8-realtime-detection\\weights\\finetuned_weights.pt')
model.to("cuda")
# model.half()

# img_test = cv2.imread("C:\\Users\\Lucas\\Desktop\\yolov8-realtime-detection\\carte.jpg")
# results = model(img_test)
# draw_boxes(img_test, results[0].boxes)

read_stream(model=model)