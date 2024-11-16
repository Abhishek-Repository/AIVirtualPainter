import cv2
import numpy as np
import os
import HandTrackingModule  # Assuming HandTrackingModule is your custom module

##################################
brush_thickness = 15
eraser_thickness = 50
##################################

# Load overlay images, including the select color image
folderPath = "Buttons"
myList = ['brush_yellow.png', 'brush_cyan.png', 'brush_green.png', 'brush_orange.png', 'brush_purple.png', 'brush_white.png', 'eraser.png', 'selection_palette.png']
overlayList = []

# Canvas Background image and reset button
folderPath2 = "extra"
reset_button = cv2.imread(os.path.join(folderPath2, "reset.png"), cv2.IMREAD_UNCHANGED)

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load image: {imPath}")
        continue
    overlayList.append(image)

select_color_overlay = overlayList[7]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandTrackingModule.HandDetector(detectionCon=0.85)
x_prev, y_prev = 0, 0
img_canvas = cv2.imread(os.path.join(folderPath2, "background_img.jpg"), cv2.IMREAD_UNCHANGED)

zone_width = 183  # Approximate width of each color zone

color_zones = {
    (0 * zone_width, 1 * zone_width): overlayList[0],
    (1 * zone_width, 2 * zone_width): overlayList[3],
    (2 * zone_width, 3 * zone_width): overlayList[5],
    (3 * zone_width, 4 * zone_width): overlayList[2],
    (4 * zone_width, 5 * zone_width): overlayList[4],
    (5 * zone_width, 6 * zone_width): overlayList[1],
    (6 * zone_width, 7 * zone_width): overlayList[6],
}

colors = [
    (89, 222, 255),  # Yellow
    (77, 145, 255),  # Cyan
    (217, 217, 217),  # Light Gray
    (87, 217, 126),  # Green
    (196, 102, 255),  # Purple
    (223, 192, 12),  # Orange
    (1, 1, 1)  # Eraser color
]

header = overlayList[0]
drawColor = colors[0]

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        ind_x, ind_y = lmList[8][1], lmList[8][2]
        mid_x, mid_y = lmList[12][1], lmList[12][2]

        fingers = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 1:
            x_prev, y_prev = 0, 0
            print("Selection Mode")

            # Show the select color overlay on the left side
            x_offset = 0
            y_offset = 0
            y1, y2 = y_offset, y_offset + select_color_overlay.shape[0]
            x1, x2 = x_offset, x_offset + select_color_overlay.shape[1]

            if select_color_overlay.shape[2] == 4:
                overlay_bgr = select_color_overlay[:, :, :3]
                overlay_alpha = select_color_overlay[:, :, 3] / 255.0
            else:
                overlay_bgr = select_color_overlay
                overlay_alpha = np.ones(overlay_bgr.shape[:2], dtype=np.float32)

            for c in range(3):
                img[y1:y2, x1:x2, c] = (
                        overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * img[y1:y2, x1:x2, c]
                )

            # Check if finger is over a specific color zone
            for i, ((x_start, x_end), brush_image) in enumerate(color_zones.items()):
                if x_start < ind_x < x_end and ind_y < select_color_overlay.shape[0]:
                    header = brush_image
                    drawColor = colors[i]
                    break

            # Display the reset button above the brush in selection mode
            reset_x_offset, reset_y_offset = img.shape[1] - header.shape[1], img.shape[0] - header.shape[0] - reset_button.shape[0]
            reset_y1, reset_y2 = reset_y_offset, reset_y_offset + reset_button.shape[0]
            reset_x1, reset_x2 = reset_x_offset, reset_x_offset + reset_button.shape[1]

            if reset_button.shape[2] == 4:
                reset_overlay_bgr = reset_button[:, :, :3]
                reset_overlay_alpha = reset_button[:, :, 3] / 255.0
            else:
                reset_overlay_bgr = reset_button
                reset_overlay_alpha = np.ones(reset_overlay_bgr.shape[:2], dtype=np.float32)

            for c in range(3):
                img[reset_y1:reset_y2, reset_x1:reset_x2, c] = (
                    reset_overlay_alpha * reset_overlay_bgr[:, :, c] +
                    (1 - reset_overlay_alpha) * img[reset_y1:reset_y2, reset_x1:reset_x2, c]
                )

            # Check if finger is over reset button
            if reset_x1 < ind_x < reset_x2 and reset_y1 < ind_y < reset_y2:
                print("Resetting canvas...")
                img_canvas = cv2.imread(os.path.join(folderPath2, "background_img.jpg"), cv2.IMREAD_UNCHANGED)

        elif fingers[1] == 1 and fingers[2] == 0:
            if drawColor:
                if x_prev == 0 and y_prev == 0:
                    x_prev, y_prev = ind_x, ind_y
                if drawColor == (1, 1, 1):
                    cv2.line(img, (x_prev, y_prev), (ind_x, ind_y), drawColor, eraser_thickness)
                    cv2.line(img_canvas, (x_prev, y_prev), (ind_x, ind_y), drawColor, eraser_thickness)
                else:
                    cv2.line(img, (x_prev, y_prev), (ind_x, ind_y), drawColor, brush_thickness)
                    cv2.line(img_canvas, (x_prev, y_prev), (ind_x, ind_y), drawColor, brush_thickness)

                x_prev, y_prev = ind_x, ind_y

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, img_canvas)

    # Overlay the selected brush icon at bottom right of the screen
    x_offset = img.shape[1] - header.shape[1]
    y_offset = img.shape[0] - header.shape[0]
    y1, y2 = y_offset, y_offset + header.shape[0]
    x1, x2 = x_offset, x_offset + header.shape[1]

    if header.shape[2] == 4:
        overlay_bgr = header[:, :, :3]
        overlay_alpha = header[:, :, 3] / 255.0
    else:
        overlay_bgr = header
        overlay_alpha = np.ones(overlay_bgr.shape[:2], dtype=np.float32)

    for c in range(3):
        img[y1:y2, x1:x2, c] = (
                overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * img[y1:y2, x1:x2, c]
        )

    if img_canvas.shape[2] == 4:
        img_canvas = img_canvas[:, :, :3]
    img_canvas = img_canvas.astype(np.uint8)

    cv2.imshow("img", img)
    cv2.imshow("Canvas", img_canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
