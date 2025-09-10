import cv2
import numpy as np

# Constants
PARAMS_WINDOW_NAME = "Parametros"
THRESHOLD_NAME = "Umbral" # Threshold for binary image
DISTANCE_NAME = "Distancia de coincidencia" # Minimum match distance 
AREA_NAME = "Area" # Minimum area to consider a contour

SQUARE = "cuadrado"
CIRCLE = "círculo"
TRIANGLE = "triángulo"

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

color_dict = {
    SQUARE: BLUE,
    CIRCLE: GREEN,
    TRIANGLE: YELLOW
}

# Frame config
frame_width = 640
frame_height = 480
cam = cv2.VideoCapture(0)
cam.set(3, frame_width)
cam.set(4, frame_height)

def do_nothing(cur):
    pass

# Adjustable parameters
cv2.namedWindow(PARAMS_WINDOW_NAME)
cv2.resizeWindow(PARAMS_WINDOW_NAME, 640, 240)
cv2.createTrackbar(THRESHOLD_NAME, PARAMS_WINDOW_NAME, 93, 255, do_nothing)
cv2.createTrackbar(DISTANCE_NAME, PARAMS_WINDOW_NAME, 20, 100, do_nothing)
cv2.createTrackbar(AREA_NAME, PARAMS_WINDOW_NAME, 500, 10000, do_nothing)

# dictionary with the contours for each figure
contour_reference = {}

# Reference image acquisition
def get_references(filename: str, figname: str) -> tuple:
    raw_figure = cv2.imread(filename)
    gray_figure = cv2.cvtColor(raw_figure, cv2.COLOR_BGR2GRAY)
    binary_figure = cv2.threshold(gray_figure, 50, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary_figure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

contour_reference[SQUARE] = get_references("square.png", SQUARE)
contour_reference[CIRCLE] = get_references("circle.jpg", CIRCLE)
contour_reference[TRIANGLE] = get_references("triangle.jpg", TRIANGLE)

# Process the image by:
def process_image(img, threshold, match_dist, min_area):
    # Turn the image to gray scale
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Turn the gray scale image to binary
    img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    # Get contours
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour found
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        best_match = None
        best_dist = float('inf')

        # Compare with reference contours
        for shape, ref_contours in contour_reference.items():
            for ref_contour in ref_contours:
                dist = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                if dist < best_dist:
                    best_dist = dist
                    best_match = shape

        color = RED # By default, red for no match
        label = "Unknown"

        if best_dist < match_dist:
            color = color_dict[best_match] # Green for an identified match
            label = best_match

        cv2.drawContours(img, [contour], -1, color, 3)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return img, img_morph

while True:
    # Read parameters
    threshold = cv2.getTrackbarPos(THRESHOLD_NAME, PARAMS_WINDOW_NAME)
    match_distance = cv2.getTrackbarPos(DISTANCE_NAME, PARAMS_WINDOW_NAME) / 100.0
    min_area = cv2.getTrackbarPos(AREA_NAME, PARAMS_WINDOW_NAME)

    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        break

    # Process the image
    img_processed, img_morph = process_image(frame.copy(), threshold, match_distance, min_area)

    cv2.imshow("Original", frame)
    cv2.imshow("Processed", img_processed)
    cv2.imshow("Morphological", img_morph)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break