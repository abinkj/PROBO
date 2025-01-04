import os
from ultralytics import YOLO
import cv2
import time
import RPi.GPIO as GPIO
import numpy as np

# Load the YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train11', 'weights', 'last.pt')
model = YOLO(model_path)

# Webcam setup
cap = cv2.VideoCapture(0)
# Motor Setup
in1_motor1 = 17  # GPIO pin for motor 1 input 1
in2_motor1 = 18  # GPIO pin for motor 1 input 2
en_motor1 = 27   # GPIO pin for motor 1 enable

in1_motor2 = 23  # GPIO pin for motor 2 input 1
in2_motor2 = 24  # GPIO pin for motor 2 input 2
en_motor2 = 25   # GPIO pin for motor 2 enable
# Motor setup
GPIO.setmode(GPIO.BCM)  
GPIO.setup(in1_motor1, GPIO.OUT)
GPIO.setup(in2_motor1, GPIO.OUT)
GPIO.setup(en_motor1, GPIO.OUT)
p_motor1 = GPIO.PWM(en_motor1, 1000)
p_motor1.start(0)  # Start with 0% duty cycle

GPIO.setup(in1_motor2, GPIO.OUT)
GPIO.setup(in2_motor2, GPIO.OUT)
GPIO.setup(en_motor2, GPIO.OUT)
p_motor2 = GPIO.PWM(en_motor2, 1000)
p_motor2.start(0)  # Start with 0% duty cycle

threshold = 0.1
tolerance = 0.1
waste_class_id = 0  # Assuming waste class is the first class

# Define red box dimensions
box_width = 50
box_height = 200

def move_robot(angle_to_turn):
    global tolerance

    print("Angle to turn:", angle_to_turn)

    if abs(angle_to_turn) < tolerance:
        # Stop both motors
        p_motor1.ChangeDutyCycle(0)
        p_motor2.ChangeDutyCycle(0)
        print("Stop")
    else:
        # Gradually adjust motor speed based on angle to turn
        max_duty_cycle = min(abs(angle_to_turn) * 2, 50)  # Limit maximum duty cycle to 50%
        if angle_to_turn > 0:
            # Turn right
            for duty_cycle in range(1, max_duty_cycle + 1):
                p_motor1.ChangeDutyCycle(duty_cycle)
                p_motor2.ChangeDutyCycle(0)
                time.sleep(0.1)  # Delay to adjust
            print("Move Right")
        else:
            # Turn left
            for duty_cycle in range(1, max_duty_cycle + 1):
                p_motor1.ChangeDutyCycle(0)
                p_motor2.ChangeDutyCycle(duty_cycle)
                time.sleep(0.1)  # Delay to adjust
            print("Move Left")

def calculate_angle(x_deviation):
    return np.arctan(x_deviation) * 180 / np.pi

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from camera")
            break

        frame_center_x = None  # Reset frame_center_x for each iteration

        # Perform object detection
        results = model(frame)[0]

        # Track waste object with highest confidence score
        max_score = 0
        max_score_box = None
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if class_id == waste_class_id and score > threshold:
                if score > max_score:
                    max_score = score
                    max_score_box = result

        if max_score_box is not None:
            x1, y1, x2, y2, score, class_id = max_score_box
            x_diff = x2 - x1
            obj_x_center = x1 + (x_diff / 2)

            # Calculate deviation from frame center
            frame_center_x = frame.shape[1] / 2

            x_deviation = obj_x_center - frame_center_x

            # Calculate angle to turn based on waste position
            angle_to_turn = calculate_angle(x_deviation)
            # Move the robot based on the calculated angle
            move_robot(angle_to_turn)

            # Determine if waste is inside red boxes
            if frame_center_x is not None and frame_center_x - box_width < obj_x_center < frame_center_x + box_width:
                # Waste is within the red boxes, stop the robot
                p_motor1.ChangeDutyCycle(0)
                p_motor2.ChangeDutyCycle(0)
                print("Waste inside red boxes, stopping...")

        else:
            # No waste object detected, stop the robot
            p_motor1.ChangeDutyCycle(0)
            p_motor2.ChangeDutyCycle(0)
            print("No waste object detected, stopping...")

        # Draw bounding boxes and display the frame with detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                # Add class name
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

        # Draw red boxes on both sides of the frame center
        if frame_center_x is not None:
            cv2.rectangle(frame, (int(frame_center_x) - box_width, 0),
                          (int(frame_center_x), frame.shape[0]), (0, 0, 255), -1)
            cv2.rectangle(frame, (int(frame_center_x), 0),
                          (int(frame_center_x) + box_width, frame.shape[0]), (0, 0, 255), -1)

        # Display the frame
        cv2.imshow('YOLO Object Detection', frame)

        # Check for keyboard interrupt ('q' key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("An error occurred:", str(e))

finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    # Cleanup GPIO
    GPIO.cleanup()