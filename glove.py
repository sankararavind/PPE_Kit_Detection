from ultralytics import YOLO
import cv2
import math
import threading  # For playing and stopping alert sound
import os
from playsound import playsound  # Import for playing sound

# Function to play alert sound in a loop
def play_alert():
    while alert_playing:
        playsound("alert.wav", block=True)

# Choose video or camera
choice = input("Enter '1' for Camera or '2' for Video: ")

if choice == '1':
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height
elif choice == '2':
    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit()
    cap = cv2.VideoCapture(video_path)  # Open video file
else:
    print("Invalid choice! Exiting...")
    exit()

# Load the YOLO model
model = YOLO("Glove_yolo.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'NO-Gloves','Person', 'Safety Cone','Safety Vest', 'machinery', 'vehicle']

alert_playing = False  # Flag to check if the alert sound is playing
alert_thread = None  # Thread for playing the alert sound

# Check if camera or video is opened
if not cap.isOpened():
    print("Error: Camera/Video not found or could not be opened.")
    exit()

try:
    while True:
        success, img = cap.read()

        if not success:
            print("Failed to capture image or end of video.")
            break

        results = model(img, stream=True)
        alert_triggered = False  # Flag to check if alert needs to be triggered

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Determine if violation or compliance
                is_violation = "NO-" in class_name
                color = (0, 0, 255) if is_violation else (0, 255, 0)  # Red for violation, Green for compliance

                # Adjust bounding box for gloves
                if class_name in ['NO-Gloves']:
                    # Focus on a smaller area in the center of the detection
                    glove_y1 = y1 + int(h * 0.6)  # Start from the lower 60%
                    glove_h = int(h * 0.25)       # Cover 25% vertically
                    glove_center_x = x1 + int(w * 0.5)  # Center X of the original box
                    glove_w = int(w * 0.3)        # Reduce width to 30% of original
                    glove_x1 = glove_center_x - int(glove_w / 2)
                    glove_x2 = glove_x1 + glove_w

                    # Draw smaller bounding box for gloves
                    cv2.rectangle(img, (glove_x1, glove_y1), (glove_x2, glove_y1 + glove_h), color, thickness=3)
                    cv2.putText(img, f'{class_name} {conf:.2f}', (glove_x1, glove_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # Draw bounding box for other classes
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=3)
                    # Add text with dynamic color
                    text_size = cv2.getTextSize(f'{class_name} {conf:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_w, text_h = text_size[0], text_size[1]
                    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), (0, 0, 0), -1)  # Black background
                    cv2.putText(img, f'{class_name} {conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Trigger alert if necessary
                if is_violation:
                    alert_triggered = True

        # Handle alert sound
        if alert_triggered and not alert_playing:
            alert_playing = True
            alert_thread = threading.Thread(target=play_alert)
            alert_thread.start()
        elif not alert_triggered and alert_playing:
            alert_playing = False
            if alert_thread:
                alert_thread.join()  # Wait for the thread to finish

        # Display the image with bounding boxes
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break
finally:
    alert_playing = False  # Stop the alert sound
    if alert_thread:
        alert_thread.join()  # Ensure the alert sound thread stops
    cap.release()  # Release the camera or video when done
    cv2.destroyAllWindows()  # Close all OpenCV windows
    