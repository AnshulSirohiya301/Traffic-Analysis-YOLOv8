import cv2
import numpy as np
import os

# --- PHASE 1.2: Video Ingestion ---
video_path = os.path.join(os.path.dirname(__file__), '..', 'traffic_video.mp4')

# Helper function to print coordinates when you click
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Coordinate: ({x}, {y})")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file. Check the filename.")
    exit()

print("Starting video stream... Press 'q' to stop.")

# Create the window beforehand so we can attach the mouse listener
cv2.namedWindow("Raw Traffic Feed")
cv2.setMouseCallback("Raw Traffic Feed", click_event)

while True:
    # Extract the next frame from the video
    success, frame = cap.read()
    
    # If success is False, the video has ended
    if not success:
        print("End of video stream.")
        break

    # Standardize resolution immediately to manage system resources
    frame = cv2.resize(frame, (1020, 500))
    
    # --- PHASE 1.3: Region of Interest (ROI) Setup ---
    # REPLACE this array with the 4 coordinates you printed out in the last step
    roi_points = np.array([[(100, 500), (300, 200), (700, 200), (900, 500)]], dtype=np.int32)
    
    # Create a completely black mask the exact same size as the frame
    mask = np.zeros_like(frame)
    
    # Draw a solid white polygon over the road area
    cv2.fillPoly(mask, roi_points, (255, 255, 255))
    
    # Merge the frame and the mask. Only pixels inside the white polygon will remain visible.
    masked_frame = cv2.bitwise_and(frame, mask)

    # Display the frames on your screen
    cv2.imshow("Raw Traffic Feed", frame)
    cv2.imshow("Isolated Road (Masked)", masked_frame)

    # Wait 30ms before reading the next frame. Press 'q' to break the loop.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up memory and destroy windows
cap.release()
cv2.destroyAllWindows()