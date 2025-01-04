import os
from ultralytics import YOLO
import cv2

video_path = r'C:\Users\abink\Desktop\Probo first train\2nddatabase\data\vv.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

model_path = os.path.join('.', 'runs', 'detect', 'train11', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.1

while ret:
 
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            #Add class name
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

           
            # Print coordinates
            print(f"Object: {results.names[int(class_id)]}, x: {int(x1)}, y: {int(y1)}")

    cv2.imshow('YOLO Object Detection', frame)
    
    # Wait for 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()  
