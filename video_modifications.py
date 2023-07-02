import cv2
import numpy as np

def blend_multiply(img1, img2):
    # Ensure the images are the same size
    assert img1.shape == img2.shape, "Both images must have the same shape."
    # Perform the multiply blend
    return (img1 * img2 / 255).astype(np.uint8)


def add_noise(img, intensity):
    # Generate a noise image
    noise = np.random.normal(0, intensity, img.shape)
    noise_img = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))  # Normalize to range 0-1
    noise_img = (noise_img * 255).astype(np.uint8)  # Scale to range 0-255
    # Blend the noise with the original image using the multiply blending mode
    img_noise = blend_multiply(img, noise_img)
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

def grayscale_image(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale

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

def rotate_image(img, angle=90):
    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy

    rotated = cv2.warpAffine(img, M, (nw, nh))
    return rotated

def edge_detection(img, low_threshold=100, high_threshold=200):
    # Apply Canny edge detection
    edges = cv2.Canny(img, low_threshold, high_threshold)
    # Dilate the edges to make them more pronounced
    dilated_edges = cv2.dilate(edges, None)
    return dilated_edges

def invert_colors(img):
    return cv2.bitwise_not(img)