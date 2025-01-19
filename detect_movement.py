import cv2
import datetime
import time

# URL of the video feed
video_url = 'http://rpiz2-2:5000/video_feed'

RECORDING_TIMEOUT = 5

# Open the video feed
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Convert the first frame to grayscale and blur it
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False
last_movement_time = 0

while True:
    ret, frame2 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the current frame to grayscale and blur it
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute the absolute difference between the current frame and the first frame
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add timestamp at the top right of the frame
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame2, timestamp, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    movement_detected = False
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        # Ignore contours with small area
        if contour_area < 500:
            continue

        # Compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Only detect movement if 'y' position is greater than 270
        if y > 270:
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            movement_detected = True

    # Start recording if movement is detected
    if movement_detected:
        if not recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(f'movement_{timestamp}.avi', fourcc, 20.0, (frame2.shape[1], frame2.shape[0]))
            recording = True
        last_movement_time = time.time()

    # Write the frame to the video file
    if recording:
        out.write(frame2)

    # Stop recording if no movement is detected for 5 seconds
    if not movement_detected and recording:
        if time.time() - last_movement_time > RECORDING_TIMEOUT:
            out.release()
            recording = False

    # Display the frame with the detected movement
    cv2.imshow('Video Feed', frame2)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the first frame
    gray1 = gray2

# Release the video capture object and close all OpenCV windows
if recording:
    out.release()
cap.release()
cv2.destroyAllWindows()