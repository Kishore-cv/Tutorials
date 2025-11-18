import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.title("IMAGE")
    plt.axis('off')
    plt.show()

def show_image_comparison(images_with_titles):
    num_images = len(images_with_titles)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]
    for i, (img, title) in enumerate(images_with_titles):
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def fs_erosion(binary_img, kernel):
    if kernel.shape[0] % 2 == 0 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be odd-sized square")
    k = kernel.shape[0]
    pad = (k - 1) // 2
    h, w = binary_img.shape[:2]
    padded = cv.copyMakeBorder(binary_img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
    output = np.zeros_like(binary_img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k, j:j+k]
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def fs_dilation(binary_img, kernel):
    if kernel.shape[0] % 2 == 0 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("Kernel must be odd-sized square")
    k = kernel.shape[0]
    pad = (k - 1) // 2
    h, w = binary_img.shape[:2]
    padded = cv.copyMakeBorder(binary_img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
    output = np.zeros_like(binary_img)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+k, j:j+k]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def fs_opening(binary_img, kernel):
    return fs_dilation(fs_erosion(binary_img, kernel), kernel)

def fs_closing(binary_img, kernel):
    return fs_erosion(fs_dilation(binary_img, kernel), kernel)


def main(image_path):

    img_bgr = cv.imread(image_path)


    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
    _, thresh = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel_3x3 = np.ones((3,3), np.uint8)
    kernel_5x5 = np.ones((5,5), np.uint8)
    kernel_7x7 = np.ones((7,7), np.uint8)

    erosion_custom = fs_erosion(thresh.copy(), kernel_3x3)
    dilation_custom = fs_dilation(thresh.copy(), kernel_3x3)
    opening_custom = fs_opening(thresh.copy(), kernel_3x3)
    closing_custom = fs_closing(thresh.copy(), kernel_5x5)

    cleaned = fs_opening(thresh.copy(), kernel_3x3)
    cleaned = fs_erosion(cleaned, kernel_3x3)
    cleaned = fs_closing(cleaned, kernel_5x5)



    show_image_comparison([
        (thresh, "Thresholded (Inverted)"),
        (erosion_custom, "Custom Erosion 3x3"),
        (dilation_custom, "Custom Dilation 3x3"),
        (opening_custom, "Custom Opening 3x3"),
        (closing_custom, "Custom Closing 5x5")
    ])

    show_image_comparison([
        (thresh, "Before Cleaning"),
        (cleaned, "After cleaning)")
    ])



main('/home/hp/Training/Opencv/Day 7/hand.jpg')

print("Day 7 Completed - Custom Morphology from Scratch + Handwritten Digits Detected Successfully")