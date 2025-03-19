import cv2
import time
from custom_select_roi import custom_select_roi

num_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
if num_cuda_devices > 0:
    print(f"CUDA is available. Number of CUDA-enabled devices: {num_cuda_devices}")
else:
    print("No CUDA-enabled devices found. Running on CPU.")

screen_size = (1920, 1080)

class SOTObjectTracker:
    def __init__(self):

        self.prev_frame_time = 0
        self.new_frame_time = time.time()

        self.tracker = None
        self.tracker_name = None

        #initialize the capture
        # /home/adak/Desktop/AcemSolutions/acem_tracker/Videos
        self.video_path = "./videos/light/light_mountainrode.mp4"
        self.cap = cv2.VideoCapture(self.video_path)

        self.track_object()

    def select_object(self, frame):
        # --- Manual ROI Selection ---
        # Press SPACE or ENTER to confirm the selection, 'c' to cancel.
        roi = custom_select_roi("Select ROI", frame)
        cv2.destroyWindow("Select ROI")

        # self.tracker = cv2.legacy.TrackerMOSSE_create()  # Very fast, lightweight; may struggle with scale/rotation changes.
        # self.tracker_name = "MOSSE"

        # self.tracker = cv2.legacy.TrackerKCF_create()    # Fast with kernelized correlation; moderate robustness.
        # self.tracker_name = "KCF"

        self.tracker = cv2.legacy.TrackerCSRT_create()  # More robust to scale, rotation and occlusion; slightly slower.
        self.tracker_name = "CSRT"

        # Initialize the tracker with the selected ROI (Only if a ROI is selected!)
        if roi:
            ok = self.tracker.init(frame, roi)
            if not ok:
                print("Error: Tracker initialization failed.")
                exit()

    def track_object(self):
        # --- Tracking Loop ---
        while True:
            self.new_frame_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                break

            # Update tracker and get updated bounding box.
            if (self.tracker):
                ok, bbox = self.tracker.update(frame)

                if ok:
                    # Tracking success: draw the tracked ROI as a rectangle.
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, "Tracking", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    # Tracking failure: display failure message.
                    cv2.putText(frame, "Lost Track", (5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                # No tracker present: display info message.
                cv2.putText(frame, "No tracker present", (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            # Display current FPS to the User.
            current_fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.new_frame_time
            cv2.putText(frame, f"{self.tracker_name} - FPS: {current_fps:.2f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 180, 0), 2)

            cv2.imshow("Tracker", cv2.resize(frame, screen_size))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press ESC to exit.
                break
            elif key == ord('l'):
                self.select_object(frame)
            elif key == ord("k"):
                self.tracker = None
                self.tracker_name = None




if __name__ == "__main__":
    SOTObjectTracker()





