from ultralytics import YOLO
import cv2
import cvzone
import time

# Set the video path or camera source
video_path = "./Videos/ppe-1.mp4"  # Replace with 0 for webcam
use_camera = video_path == "0"  # Detect if a webcam is being used

# Open video or camera feed
cap = cv2.VideoCapture(0 if use_camera else video_path)


# Check if the video/camera feed is valid
if not cap.isOpened():
    print(f"Error: Could not open {'camera' if use_camera else 'video file'} at {video_path}.")
    exit()

# Set resolution for video capture (if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load the YOLO model
model_path = "best.pt"  # Ensure the path to the trained model is correct
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    cap.release()
    exit()

# Define class names (corresponding to your trained model)
classNames = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
    'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle'
]

# Initialize FPS calculation variables
prev_frame_time = 0

print("Press 'q' to exit.")
while True:
    # Read a frame from the video/camera
    success, img = cap.read()

    # Break the loop if no frame is captured
    if not success:
        print("No frame captured. Exiting...")
        break

    # Run the YOLO model on the frame
    try:
        results = model(img, stream=True)
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        break

    # Process each detection result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            w, h = x2 - x1, y2 - y1

            # Draw bounding box with corner rectangle
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 255, 0), thickness=2)

            # Extract confidence score and class ID
            conf = round(box.conf[0] * 100, 2)
            cls = int(box.cls[0])

            # Get class name
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"

            # Draw the class label and confidence score
            label = f"{class_name} {conf}%"
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

    # Display FPS on the frame
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    # Show the frame
    cv2.imshow("Detection", img)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
