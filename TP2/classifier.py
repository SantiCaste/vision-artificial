import cv2
import numpy as np
import dataset_generator as dg
from joblib import load

# Process the image by:
def process_image(img, threshold, min_area):
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
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            valid_contours.append(contour)
            
    
        cv2.drawContours(img, [contour], -1, dg.GREEN, 3)

    return img, img_morph, valid_contours


def predict_shape(contours, classifier):
    # prediction = classifier.predict([hu_moments_log])[0]
    # proba = classifier.predict_proba([hu_moments_log])[0]
    # confidence = np.max(proba)
    # if confidence < match_distance:
    #     return None, confidence
    # return dg.color_dict[prediction], confidence
    for contour in contours:
        hu_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
        prediction = classifier.predict([hu_moments])
        print(f"prediction: {prediction}")
        label = prediction[0]
        # print(f"Predicted shape: {label}")
        color = dg.color_dict.get(label, dg.RED)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img_processed, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    

if __name__ == "__main__":
    # Load the pre-trained model

    frameWidth = 640
    frameHeight = 480 
    cam = cv2.VideoCapture(0)
    cam.set(3, frameWidth)
    cam.set(4, frameHeight)

    classifier = load('shape_classifier.joblib')
    cv2.namedWindow(dg.PARAMS_WINDOW_NAME)
    cv2.resizeWindow(dg.PARAMS_WINDOW_NAME, 640, 240)
    cv2.createTrackbar(dg.THRESHOLD_NAME, dg.PARAMS_WINDOW_NAME, 93, 255, dg.do_nothing)
    cv2.createTrackbar(dg.AREA_NAME, dg.PARAMS_WINDOW_NAME, 500, 10000, dg.do_nothing)

    while True:
        # Read parameters
        threshold = cv2.getTrackbarPos(dg.THRESHOLD_NAME, dg.PARAMS_WINDOW_NAME)
        min_area = cv2.getTrackbarPos(dg.AREA_NAME, dg.PARAMS_WINDOW_NAME)

        # Capture frame-by-frame
        ret, frame = cam.read()
        if not ret:
            continue

        # Process the image
        img_processed, img_morph, valid_contours = process_image(frame.copy(), threshold, min_area)
        cv2.imshow("Original", frame)
        cv2.imshow("Processed", img_processed)
        cv2.imshow("Morphological", img_morph)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("Esc key pressed. Exiting...")
            break

        if valid_contours:
            predict_shape(valid_contours, classifier)