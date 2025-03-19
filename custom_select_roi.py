import cv2

def custom_select_roi(window_name, image):
    """
    Displays an image in a window and lets the user draw an ROI.
    Returns the selected ROI as (x, y, w, h) when Enter is pressed.
    Press 'c' to cancel and return None.
    """
    roi = None
    cropping = False
    start_point = (0, 0)
    end_point = (0, 0)
    orig = image.copy()   # Original image
    clone = orig.copy()   # Clone for drawing (will reset on new draw)

    def mouse_callback(event, x, y, flags, param):
        nonlocal cropping, start_point, end_point, clone, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            # Reset the drawing if starting a new selection
            cropping = True
            start_point = (x, y)
            end_point = (x, y)
            clone = orig.copy()  # Clear any previous ROI drawing
            cv2.imshow(window_name, clone)
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                end_point = (x, y)
                temp_image = clone.copy()
                # Draw a thin (1px) rectangle for live feedback
                cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 1)
                cv2.imshow(window_name, temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            cropping = False
            end_point = (x, y)
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            roi = (x1, y1, x2 - x1, y2 - y1)
            clone = orig.copy()  # Reset clone before drawing the final ROI
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow(window_name, clone)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        # Press Enter (ASCII 13) to confirm the ROI selection.
        if key == 13:
            break
        # Press 'c' (ASCII) to cancel the selection.
        elif key == ord('c'):
            roi = None
            break

    cv2.destroyWindow(window_name)
    return roi

# Example usage:
if __name__ == '__main__':
    img = cv2.imread("your_image.jpg")  # Replace with your image path
    selected_roi = custom_select_roi("Custom ROI Selector", img)
    if selected_roi:
        print("ROI selected:", selected_roi)
    else:
        print("ROI selection cancelled.")
