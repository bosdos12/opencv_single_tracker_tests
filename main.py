import cv2
import time

# --- Choose Your Tracker ---
# Uncomment only one of the following lines to select the tracker.

# tracker = cv2.legacy.TrackerMOSSE_create()  # Very fast, lightweight; may struggle with scale/rotation changes.
# tracker_name = "MOSSE"

tracker = cv2.legacy.TrackerKCF_create()    # Fast with kernelized correlation; moderate robustness.
tracker_name = "KCF"

# tracker = cv2.legacy.TrackerCSRT_create()       # More robust to scale, rotation and occlusion; slightly slower.
# tracker_name = "CSRT"

# --- Video Capture Setup ---
# Change video_path to 0 for webcam or a path to your video file.
video_path = "./12430396-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

prev_frame_time = 0

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame.
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# --- Manual ROI Selection ---
# Let the user draw a square ROI on the first frame.
# Press SPACE or ENTER to confirm the selection, 'c' to cancel.
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Initialize the tracker with the selected ROI.
ok = tracker.init(frame, roi)
if not ok:
    print("Error: Tracker initialization failed.")
    cap.release()
    exit()

# --- Tracking Loop ---
while True:
    new_frame_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker and get updated bounding box.
    ok, bbox = tracker.update(frame)

    if ok:
        # Tracking success: draw the tracked ROI as a rectangle.
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Tracking failure: display failure message.
        cv2.putText(frame, "Lost Track", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


    # Display current FPS to the User.
    current_fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"{tracker_name} - FPS: {current_fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)


    cv2.imshow("Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit.
        break

cap.release()
cv2.destroyAllWindows()
