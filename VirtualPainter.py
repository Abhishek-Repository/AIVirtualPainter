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
# print(myList)
overlayList = []

# Canvas Background image
folderPath2 = "extra"

for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load image: {imPath}")
        continue
    overlayList.append(image)
# overlayList contains: ['brush_black.png', 'brush_cyan.png', 'brush_green.png', 'brush_orange.png', 'brush_purple.png', 'brush_white.png', 'eraser.png', 'selection_palette.png']

# Load select_color.png for color selection overlay
select_color_overlay = overlayList[7]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandTrackingModule.HandDetector(detectionCon=0.85)
x_prev, y_prev = 0, 0
# Creating a new canvas to draw
img_canvas = cv2.imread(os.path.join(folderPath2, "background_img.jpg"), cv2.IMREAD_UNCHANGED)
print(img_canvas.shape)

# Define x-coordinates for each color zone based on the selection_palette image
zone_width = 183  # Approximate width of each color zone

color_zones = {
    (0 * zone_width, 1 * zone_width): overlayList[0],  # black
    (1 * zone_width, 2 * zone_width): overlayList[3],  # orange
    (2 * zone_width, 3 * zone_width): overlayList[5],  # gray
    (3 * zone_width, 4 * zone_width): overlayList[2],  # green
    (4 * zone_width, 5 * zone_width): overlayList[4],  # pink
    (5 * zone_width, 6 * zone_width): overlayList[1],  # cyan
    (6 * zone_width, 7 * zone_width): overlayList[6],  # eraser
}

colors = [
    (89, 222, 255),  # Yellow (255, 222, 89)
    (77, 145, 255),  # RGB (255, 145, 77) -> BGR (77, 145, 255)
    (217, 217, 217),  # Light Gray (unchanged)
    (87, 217, 126),  # RGB (126, 217, 87) -> BGR (87, 217, 126)
    (196, 102, 255),  # RGB (255, 102, 196) -> BGR (196, 102, 255)
    (223, 192, 12),  # RGB (12, 192, 223) -> BGR (223, 192, 12)
    (1, 1, 1)  # No color for eraser
]

# Default selected brush and color
header = overlayList[0]
drawColor = colors[0]  # Set to black initially

while True:
    # 1. Capture image
    success, img = cap.read()
    if not success:
        break

    # Flip the frame to correct mirroring
    img = cv2.flip(img, 1)

    # 2. Find the hand landmarks
    # 2. Find the hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:   #Whenever hand is detected
        # 3. Get the tip coordinates of index and middle fingers
        ind_x, ind_y = lmList[8][1], lmList[8][2]
        mid_x, mid_y = lmList[12][1], lmList[12][2]

        # 4. Check which fingers are up
        fingers = detector.fingersUp()

        # 5. If Selection mode - Two fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            x_prev, y_prev = 0, 0
            print("Selection Mode")

            # Show the select color overlay on the left side
            x_offset = 0  # Set to left of the screen
            y_offset = 0
            y1, y2 = y_offset, y_offset + select_color_overlay.shape[0]
            x1, x2 = x_offset, x_offset + select_color_overlay.shape[1]

            # Separate BGR and alpha channel of select color overlay
            if select_color_overlay.shape[2] == 4:
                overlay_bgr = select_color_overlay[:, :, :3]
                overlay_alpha = select_color_overlay[:, :, 3] / 255.0
            else:
                overlay_bgr = select_color_overlay
                overlay_alpha = np.ones(overlay_bgr.shape[:2], dtype=np.float32)

            # Blend overlay with the region of interest on the background frame
            for c in range(3):  # Iterate over BGR channels
                img[y1:y2, x1:x2, c] = (
                        overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * img[y1:y2, x1:x2, c]
                )

            # Check if finger is over a specific color zone
            for i, ((x_start, x_end), brush_image) in enumerate(color_zones.items()):
                if x_start < ind_x < x_end and ind_y < select_color_overlay.shape[0]:
                    header = brush_image
                    drawColor = colors[i]  # Set drawColor to the corresponding color
                    break

            # Display the selected color as a rectangle on the screen
            if drawColor:
                cv2.rectangle(img, (ind_x, ind_y - 15), (mid_x, mid_y + 15), drawColor, cv2.FILLED)

        # 6. If Drawing mode - Only index finger is up
        elif fingers[1] == 1 and fingers[2] == 0:
            if drawColor:
                if x_prev == 0 and y_prev == 0:
                    x_prev, y_prev = ind_x, ind_y
                if drawColor == (1, 1, 1):
                    # cv2.imshow()
                    cv2.line(img, (x_prev, y_prev), (ind_x, ind_y), drawColor, eraser_thickness)
                    cv2.line(img_canvas, (x_prev, y_prev), (ind_x, ind_y), drawColor, eraser_thickness)
                else:
                    # cv2.circle(img, (ind_x, ind_y - 15), 15, drawColor, cv2.FILLED)
                    cv2.line(img, (x_prev, y_prev), (ind_x, ind_y), drawColor, brush_thickness)
                    cv2.line(img_canvas, (x_prev, y_prev), (ind_x, ind_y), drawColor, brush_thickness)

                x_prev, y_prev = ind_x, ind_y
            print("Drawing Mode")

    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, ImgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  #Creating inverse image to make the region black
    # imgInv = cv2.cvtColor(ImgInv, cv2.COLOR_GRAY2BGR) # We are converting back so that we can add it to original image
    # img = cv2.bitwise_and(img, imgInv)
    # img = cv2.bitwise_or(img, img_canvas)

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  #converting the gray img to image inverse
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)        # We are converting back so that we can add it to original image
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,img_canvas)


    # Overlay the selected brush image at the top right of the screen
    x_offset = img.shape[1] - header.shape[1]
    y_offset = img.shape[0] - header.shape[0]
    y1, y2 = y_offset, y_offset + header.shape[0]
    x1, x2 = x_offset, x_offset + header.shape[1]

    # Separate BGR and alpha channel of selected brush image
    if header.shape[2] == 4:
        overlay_bgr = header[:, :, :3]
        overlay_alpha = header[:, :, 3] / 255.0
    else:
        overlay_bgr = header
        overlay_alpha = np.ones(overlay_bgr.shape[:2], dtype=np.float32)

    # Blend overlay with the region of interest on the background frame
    for c in range(3):  # Iterate over BGR channels
        img[y1:y2, x1:x2, c] = (
                overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * img[y1:y2, x1:x2, c]
        )

    # Ensure `img_canvas` is in the correct format before displaying
    if img_canvas.shape[2] == 4:  # Remove alpha if it has 4 channels
        img_canvas = img_canvas[:, :, :3]
    img_canvas = img_canvas.astype(np.uint8)  # Ensure dtype is uint8

    # Display the result
    cv2.imshow("img", img)
    cv2.imshow("Canvas", img_canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
