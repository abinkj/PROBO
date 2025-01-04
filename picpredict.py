import os
from ultralytics import YOLO
import cv2

image_path = r'C:\Users\abink\Desktop\Probo first train\2nddatabase\data\Screenshot 2024-02-13 200402.png'
output_image_path = 'output_image.jpg'

# Load the input image
image = cv2.imread(image_path)

# Initialize YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
model = YOLO(model_path)

# Perform object detection on the image
results = model(image)

# Print the type and content of the results object for debugging
print("Type of results:", type(results))
print("Content of results:", results)

# Check if there are any detections
if isinstance(results, list) and len(results) > 0:
    # Get the first element of the list (assuming it contains the results)
    results = results[0]
    
    # Check if there are any detections
    if len(results.pred) == 0:
        print("No objects detected.")
    else:
        # Get the results including bounding boxes, class names, and confidence scores
        boxes = results.xyxy[0].cpu().numpy()
        class_ids = results.pred[0][:, -1].cpu().numpy().astype(int)
        confidence_scores = results.pred[0][:, 4].cpu().numpy()

        # Set a threshold for object detection confidence
        threshold = 0.5

        # Draw bounding boxes and class labels on the image
        for box, class_id, score in zip(boxes, class_ids, confidence_scores):
            if score > threshold:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(image, model.names[class_id].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Save the output image with bounding boxes and class labels
        cv2.imwrite(output_image_path, image)

        # Display the output image
        cv2.imshow('Object Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Error: No results returned from object detection.")
