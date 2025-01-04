import os
from ultralytics import YOLO
import cv2

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train5', 'weights', 'last.pt')
model = YOLO(model_path)

threshold = 0.3

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform object detection on the frame
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Add class name
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

            # Print coordinates
            print(f"Object: {results.names[int(class_id)]}, x: {int(x1)}, y: {int(y1)}")

    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', frame)
    
    # Check for keyboard interrupt ('q' key)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
