import os
from ultralytics import YOLO
import cv2
import time
import RPi.GPIO as GPIO
import numpy as np
import sys

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

# Ultrasonic sensor (HC-SR04) Setup
trigger_pin = 7  # GPIO pin for sensor trigger
echo_pin = 5     # GPIO pin for sensor echo

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

# Ultrasonic sensor setup
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

belt = 21

GPIO.setup(belt, GPIO.OUT)
def belt_off():
    GPIO.output(belt, GPIO.HIGH)
belt_off()

threshold = 0.7
tolerance = 0.1
waste_class_id = 0  # Assuming waste class is the first class

# Define red box dimensions
box_width = 50
box_height = 200

def measure_distance():
    # Trigger ultrasonic sensor
    GPIO.output(trigger_pin, GPIO.LOW)
    time.sleep(0.5)
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)

    # Measure echo duration
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    # Convert duration to distance (in cm)
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    if distance>250:
        distance=240
    return distance
def calculate_angle(x_deviation):
    return np.arctan(x_deviation) * 180 / np.pi

def move_forward(t):
    # pwm_motor1.start(1)
    # pwm_motor2.start(1)
    GPIO.output(in1_motor1, GPIO.HIGH)
    GPIO.output(in2_motor1, GPIO.LOW)
    GPIO.output(in1_motor2, GPIO.HIGH)
    # while True:
    #             distance = measure_distance()
    #             print("Distance:", distance, "cm")

    #             if distance <= 30:
    #                 # stop()
    #                 # belt_off()
    #                 break
    time.sleep(t)
    GPIO.output(in1_motor1, GPIO.LOW)
    GPIO.output(in2_motor1, GPIO.LOW)
    GPIO.output(in1_motor2, GPIO.LOW)
    GPIO.output(in1_motor2, GPIO.LOW)
    belt_on()
    time.sleep(10)
    belt_off()

def right(x,a):
    rot_t=a*0.017556/4
    if x>175:
     GPIO.output(in1_motor1, GPIO.HIGH)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.HIGH)
     time.sleep(rot_t)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(3)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
   
     move_forward(t)
     
    elif x<175 and x>150:
     GPIO.output(in1_motor1, GPIO.HIGH)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.HIGH)
     time.sleep(rot_t-0.09)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(3)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
     # Move the robot forward if waste is detected
     move_forward(t)
     
    elif x<150:
     GPIO.output(in1_motor1, GPIO.HIGH)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.HIGH)
     time.sleep(0.23)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(3)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
     # Move the robot forward if waste is detected
     move_forward(t)
     

def left(x,a):
    rot_t=a*0.017556/4
    if x < -175:
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.HIGH)
     GPIO.output(in1_motor2, GPIO.HIGH)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(rot_t)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(3)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
   
     move_forward(t)
     
    elif x > -175 and x < -150:
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.HIGH)
     GPIO.output(in1_motor2, GPIO.HIGH)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(rot_t-0.09)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(2)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
     
     move_forward(t)
   
    elif x > -150:
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.HIGH)
     GPIO.output(in1_motor2, GPIO.HIGH)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(rot_t-.22)
     GPIO.output(in1_motor1, GPIO.LOW)
     GPIO.output(in2_motor1, GPIO.LOW)
     GPIO.output(in1_motor2, GPIO.LOW)
     GPIO.output(in2_motor2, GPIO.LOW)
     time.sleep(3)
     distance = measure_distance()
     print("Distance:", distance, "cm")
     t=distance/51
     t=t-0.5
     
     move_forward(t)
     



def belt_on():
    GPIO.output(belt, GPIO.LOW)



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

        # Display the frame
       

        # Track waste object with highest confidence score
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.07:  # Filter out low-confidence detections
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Add confidence score
                cv2.putText(frame, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detected_frame = frame.copy()
        # Show the frame
        cv2.imshow('YOLO Object Detection', frame)
        cv2.waitKey(1)
        max_score = 0
        max_score_box = None
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > max_score:
                max_score = score
                max_score_box = result

        if max_score_box is not None:
            x1, y1, x2, y2, score, class_id = max_score_box
            x_diff = x2 - x1
            obj_x_center = x1 + (x_diff / 2)

            # Calculate deviation from frame center
            frame_center_x = frame.shape[1] / 2
            if frame_center_x is not None:
            #red box for frame center
               cv2.rectangle(frame, (int(frame_center_x) - box_width, 0),
                          (int(frame_center_x) + box_width, frame.shape[0]), (0, 0, 255), -1)
               cv2.imshow('YOLO Object Detection', frame)
               cv2.waitKey(1)
            x_deviation = obj_x_center - frame_center_x
            ang =abs( calculate_angle(x_deviation))
            print(ang)

            # Draw green box around detected waste
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # time.sleep(10)
            print(x_deviation)
            if x_deviation > 30:              
                right(x_deviation,ang)
                if detected_frame is not None:
                     if frame_center_x is not None:
                          cv2.rectangle(detected_frame, (int(frame_center_x) - box_width, 0),
                          (int(frame_center_x) + box_width, frame.shape[0]), (0, 0, 255), -1)
                     cv2.imshow('Detected Waste', detected_frame)
                     cv2.waitKey(0)
                sys.exit(0)
            elif x_deviation < -30:
                left(x_deviation,ang)
                if detected_frame is not None:
                     cv2.imshow('Detected Waste', detected_frame)
                     cv2.waitKey(0)
                sys.exit(0)
            else:
                distance = measure_distance()
                print("Distance:", distance, "cm")
                t=distance/51
                t=t-0.5
   
                move_forward(t)

        # Draw red box to indicate frame center
        if frame_center_x is not None:
            cv2.rectangle(frame, (int(frame_center_x) - box_width, 0),
                          (int(frame_center_x) + box_width, frame.shape[0]), (0, 0, 255), -1)

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