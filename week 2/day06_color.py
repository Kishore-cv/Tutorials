import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.title("IMAGE")
    plt.axis('off')
    plt.show()

img_bgr = cv.imread('balls.jpg')
if img_bgr is None:
    raise FileNotFoundError("Image not found! Please place 'objects_color.jpg' in the same folder.")

img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

h_channel = img_hsv[:, :, 0]
s_channel = img_hsv[:, :, 1]
v_channel = img_hsv[:, :, 2]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title('Original RGB Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(h_channel, cmap='hsv')
axs[0, 1].set_title('Hue Channel')
axs[0, 1].axis('off')

axs[0, 2].imshow(s_channel, cmap='gray')
axs[0, 2].set_title('Saturation Channel')
axs[0, 2].axis('off')

axs[1, 0].imshow(v_channel, cmap='gray')
axs[1, 0].set_title('Value Channel')
axs[1, 0].axis('off')

hist_h = cv.calcHist([h_channel], [0], None, [180], [0, 180])
axs[1, 1].plot(hist_h, color='red')
axs[1, 1].set_title('Hue Histogram')
axs[1, 1].set_xlim([0, 180])

hist_s = cv.calcHist([s_channel], [0], None, [256], [0, 256])
hist_v = cv.calcHist([v_channel], [0], None, [256], [0, 256])
axs[1, 2].plot(hist_s, color='green', label='Saturation')
axs[1, 2].plot(hist_v, color='blue', label='Value')
axs[1, 2].set_title('Saturation & Value Histograms')
axs[1, 2].legend()

plt.tight_layout()
plt.show()

lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv.inRange(img_hsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(img_hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

lower_blue = np.array([60, 0, 0])
upper_blue = np.array([110, 255, 255])
mask_blue = cv.inRange(img_hsv, lower_blue, upper_blue)

lower_green = np.array([35, 0, 0])
upper_green = np.array([65, 255, 255])
mask_green = cv.inRange(img_hsv, lower_green, upper_green)

result_red = cv.bitwise_and(img_rgb, img_rgb, mask=mask_red)
result_blue = cv.bitwise_and(img_rgb, img_rgb, mask=mask_blue)
result_green = cv.bitwise_and(img_rgb, img_rgb, mask=mask_green)

lower_yellow = np.array([15, 51, 118])
upper_yellow = np.array([32, 255, 255])
mask_yellow = cv.inRange(img_hsv, lower_yellow, upper_yellow)

result_yellow = cv.bitwise_and(img_rgb, img_rgb, mask=mask_yellow)

plt.figure(figsize=(18, 10))

plt.subplot(2, 5, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(mask_red, cmap='gray')
plt.title('Red Mask')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(mask_blue, cmap='gray')
plt.title('Blue Mask')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(mask_green, cmap='gray')
plt.title('Green Mask')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(result_red)
plt.title('Segmented Red Objects')
plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(result_blue)
plt.title('Segmented Blue Objects')
plt.axis('off')

plt.subplot(2, 5, 7)
plt.imshow(result_green)
plt.title('Segmented Green Objects')
plt.axis('off')

plt.subplot(2, 5, 8)

plt.imshow(result_yellow)
plt.title('Segmented Yellow Objects')
plt.axis('off')

plt.subplot(2, 5, 9)
combined_mask = cv.bitwise_or(cv.bitwise_or(mask_red, mask_blue), mask_green)
combined_result = cv.bitwise_and(img_rgb, img_rgb, mask=combined_mask)
plt.imshow(combined_result)
plt.title('All Colored Objects Segmented')
plt.axis('off')

plt.subplot(2, 5, 10)
combined_mask = cv.bitwise_or(combined_mask, mask_yellow)
combined_result = cv.bitwise_and(img_rgb, img_rgb, mask=combined_mask)
plt.imshow(combined_result)
plt.title('All Colored Objects Segmented')
plt.axis('off')

plt.tight_layout()
plt.show()



print("Day 6 completed: Color segmentation and histograms displayed & saved.")