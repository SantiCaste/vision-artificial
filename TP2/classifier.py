import cv2
import numpy as np
import dataset_generator as dg
from joblib import load

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
        label = prediction[0]
        color = dg.color_dict.get(label, dg.RED)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img_processed, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    

if __name__ == "__main__":
    # Load the pre-trained model
    classifier = load('shape_classifier.joblib')
    cv2.namedWindow(dg.PARAMS_WINDOW_NAME)
    cv2.resizeWindow(dg.PARAMS_WINDOW_NAME, 640, 240)
    cv2.createTrackbar(dg.THRESHOLD_NAME, dg.PARAMS_WINDOW_NAME, 93, 255, dg.do_nothing)
    cv2.createTrackbar(dg.AREA_NAME, dg.PARAMS_WINDOW_NAME, 500, 10000, dg.do_nothing)
    cv2.createTrackbar(dg.DISTANCE_NAME, dg.PARAMS_WINDOW_NAME, 20, 100, dg.do_nothing)
    

    while True:
        # Read parameters
        threshold = cv2.getTrackbarPos(dg.THRESHOLD_NAME, dg.PARAMS_WINDOW_NAME)
        match_distance = cv2.getTrackbarPos(dg.DISTANCE_NAME, dg.PARAMS_WINDOW_NAME) / 100.0
        min_area = cv2.getTrackbarPos(dg.AREA_NAME, dg.PARAMS_WINDOW_NAME)

        # Capture frame-by-frame
        ret, frame = dg.cam.read()
        if not ret:
            continue

        # Process the image
        img_processed, img_morph, valid_contours = dg.process_image(frame.copy(), threshold, min_area)

        if valid_contours:
            predict_shape(valid_contours, classifier)