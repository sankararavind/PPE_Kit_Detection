from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import os


def initialize_capture():
    """
    Initializes the video capture from a camera or a video file based on user input.
    Returns:
        cap (cv2.VideoCapture): The video capture object.
    """
    print("Enter '1' for Camera or '2' for Video:")
    choice = input().strip()

    if choice == '1':  # Use camera
        cap = cv2.VideoCapture(0)  # Default camera
        cap.set(3, 1280)  # Frame width
        cap.set(4, 720)  # Frame height
        if not cap.isOpened():
            raise ValueError("Error: Camera not found or could not be opened.")
    elif choice == '2':  # Use video file
        video_path = input("Enter the path to the video file: ").strip()
        if not os.path.exists(video_path):
            raise FileNotFoundError("Error: Video file not found!")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Unable to open the video file.")
    else:
        raise ValueError("Invalid choice! Enter '1' for Camera or '2' for Video.")
    return cap


def main():
    try:
        # Initialize the video capture
        cap = initialize_capture()

        # Load the YOLO model
        model = YOLO("best.pt")  # Ensure this path points to your YOLO weights file

        # Define class names
        classNames = [
            'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
            'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
        ]

        # Initialize variables for FPS calculation
        prev_frame_time = 0
        fps = 0

        while True:
            success, img = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break

            # Run YOLO model on the frame
            results = model(img, stream=True)
            for r in results:
                for box in r.boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 0, 0))  # Draw rectangle

                    # Confidence
                    conf = math.ceil(box.conf[0] * 100) / 100
                    cls = int(box.cls[0])  # Class ID

                    # Draw label
                    cvzone.putTextRect(
                        img,
                        f'{classNames[cls]} {conf:.2f}',
                        (max(0, x1), max(35, y1)),
                        scale=1,
                        thickness=1
                    )

            # Calculate FPS
            current_time = time.time()
            if prev_frame_time != 0:
                fps = int(1 / (current_time - prev_frame_time))
            prev_frame_time = current_time

            # Display FPS on the image
            cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)
            print(f'FPS: {fps}')

            # Show the image
            cv2.imshow("Detection", img)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Release resources
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Released all resources and closed windows.")


if __name__ == "__main__":
    main()
