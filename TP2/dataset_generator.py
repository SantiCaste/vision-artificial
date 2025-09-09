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

HU_MOMENTS_FILE = "hu_moments.csv"

color_dict = {
    SQUARE: BLUE,
    CIRCLE: GREEN,
    TRIANGLE: YELLOW
}

key_shape_mapping = {
    ord('s'): SQUARE,
    ord('c'): CIRCLE,
    ord('t'): TRIANGLE
}

def do_nothing(cur):
    pass


# get image contours
def get_contours(img, img_contour, min_area=1000):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, count in enumerate(contours):
        if i == 0:
            continue

        area =  cv2.contourArea(count)
        if area < min_area:
            continue

        perimeter  = cv2.arcLength(count, True)
        approximation = cv2.approxPolyDP(count, 0.02 * perimeter, True)
        cv2.drawContours(img_contour, contours, i, GREEN, 3)

        x, y, w, h = cv2.boundingRect(approximation)
        x_center = int((2 * x + w) / 2)
        y_center = int((2 * y + h) / 2)
        coordinates = (x_center, y_center)
        color = (0, 0, 0)

# Process the image by:
def process_image(img, threshold, min_area):
    # Turn the image to gray scale
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Turn the gray scale image to binary
    img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)[1]

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    # Get contours
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour found
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            valid_contours.append(contour)
            
    
    cv2.drawContours(img, [contour], -1, GREEN, 3)

    return img, img_morph, valid_contours

# Main loop
if __name__== "__main__":
    # Frame config
    frame_width = 640
    frame_height = 480
    cam = cv2.VideoCapture(0)
    cam.set(3, frame_width)
    cam.set(4, frame_height)
    # Adjustable parameters
    cv2.namedWindow(PARAMS_WINDOW_NAME)
    cv2.resizeWindow(PARAMS_WINDOW_NAME, 640, 240)
    cv2.createTrackbar(THRESHOLD_NAME, PARAMS_WINDOW_NAME, 93, 255, do_nothing)
    cv2.createTrackbar(DISTANCE_NAME, PARAMS_WINDOW_NAME, 20, 100, do_nothing)
    cv2.createTrackbar(AREA_NAME, PARAMS_WINDOW_NAME, 500, 10000, do_nothing)


    try:
        with open(HU_MOMENTS_FILE, "w") as f:
            tags = []
            while True:
                # Read parameters
                threshold = cv2.getTrackbarPos(THRESHOLD_NAME, PARAMS_WINDOW_NAME)
                match_distance = cv2.getTrackbarPos(DISTANCE_NAME, PARAMS_WINDOW_NAME) / 100.0
                min_area = cv2.getTrackbarPos(AREA_NAME, PARAMS_WINDOW_NAME)

                # Capture frame-by-frame
                ret, frame = cam.read()
                if not ret:
                    continue

                # Process the image
                img_processed, img_morph, valid_contours = process_image(frame.copy(), threshold, min_area)

                cv2.imshow("Original", frame)
                cv2.imshow("Processed", img_processed)
                cv2.imshow("Morphological", img_morph)

                key = cv2.waitKey(30) & 0xFF
                if key in key_shape_mapping.keys():
                    if len(valid_contours) == 1:
                        print("Error: more than one contour detected for the current frame.")
                        continue
                    
                    shape = key_shape_mapping[key]
                    print(f"Saving contour for shape: {shape}")
                    tags.append(shape)
                    moments = cv2.moments(valid_contours[0])
                    hu_moments = cv2.HuMoments(moments).flatten() # TODO: check if flatten is correct and needed
                    print(f"Hu Moments: {hu_moments}")
                    f.write(f"{shape}," + ",".join([str(h) for h in hu_moments]) + "\n")

                elif key == ord('q'):
                    print("Exiting...")
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
