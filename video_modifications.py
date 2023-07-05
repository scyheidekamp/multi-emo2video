import cv2
import numpy as np

def add_noise(img, intensity):
    # Generate noise
    noise = np.random.normal(0, 1, img.shape).astype(np.float32)
    # Scale the noise by intensity/10
    noise = noise * (intensity*2)
    # Add the noise to the image
    img_noise = img + noise
    # Clip the values to be between 0 and 255
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
    return img_noise

def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    value_uint8 = np.uint8(value)
    v[v <= lim] += value_uint8
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def desaturate_and_blur(img, intensity, factor=2):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Decrease saturation, ensuring it doesn't go below 0
    hsv[:,:,1] = np.clip(hsv[:,:,1]*(1.0 - (intensity/(factor*50))), 0, 255)
    # Convert back to BGR
    desaturated_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Blur the image
    blurred = blur_image(desaturated_bgr, intensity/factor)
    return blurred

def blur_image(img, intensity):
    # Ensure intensity is positive
    intensity = max(0.1, intensity)
    # Convert intensity to a kernel size, ensuring it's odd
    kernel_width = int(intensity * 2)
    kernel_width += 1 if kernel_width % 2 == 0 else 0
    kernel_height = int(intensity * 2)
    kernel_height += 1 if kernel_height % 2 == 0 else 0
    kernel_size = (kernel_width, kernel_height)
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    return blurred

def rotate_image(img, angle, factor=3):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    # Modify angle based on factor
    angle = angle * factor
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def edge_detection(image, sigma=0.33):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # Convert single channel image to 3-channel image
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    # return the edged image
    return edged

def invert_colors(img, emotion_intensity):
    emotion_intensity = emotion_intensity
    return cv2.bitwise_not(img)

def threshold_image(img, threshold):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    ret, thresh = cv2.threshold(gray, (threshold*2), 255, cv2.THRESH_BINARY)
    # Convert single channel image back to 3 channels.
    thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh_colored

def add_text_overlay(img, text, position=(50, 50), font_scale=1, color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def translate_image(img, shift):
    (height, width) = img.shape[:2]
    translation_matrix = np.float32([[1, 0, shift*10], [0, 1, shift*10]])
    return cv2.warpAffine(img, translation_matrix, (width, height))

def adjust_contrast(img, emotion_intensity):
    # Map the emotion intensity to the range 1 to 3
    factor = 1 + (emotion_intensity/30)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def increase_saturation(img, increase_factor):
    # Convert BGR image to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Increase saturation, make sure it doesn't exceed 255
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] * (increase_factor/2), 0, 255)
    # Convert back to BGR
    saturated_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return saturated_img

