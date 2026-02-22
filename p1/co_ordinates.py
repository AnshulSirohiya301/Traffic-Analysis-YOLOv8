import cv2
import os

roi_points = []

def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"You clicked: ({x}, {y})")
        roi_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click your 4 ROI points", frame)

video_path = os.path.join(os.path.dirname(__file__), '..', 'highway_test.mp4')
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

frame = cv2.resize(frame, (1020, 500))

print("Click 4 points to outline the drivable road.")
print("Order: Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right")
print("Press 'q' when you are done.")

cv2.imshow("Click your 4 ROI points", frame)
cv2.setMouseCallback("Click your 4 ROI points", get_coordinates)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\nCOPY THIS ARRAY INTO YOUR MAIN SCRIPT:")
print(f"roi_points = np.array([{roi_points}], dtype=np.int32)")