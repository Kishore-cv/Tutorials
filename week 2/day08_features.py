import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img):
    if len(img.shape) == 3:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Feature detection

def harris_corner_detector(img):
    corners = cv.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    corners_dilated = cv.dilate(corners, None)
    thresh = 0.01 * corners_dilated.max()
    result = img_color.copy()
    result[corners_dilated > thresh] = [0, 0, 255]
    kp_count = np.sum(corners_dilated > thresh)
    return result, kp_count




def shi_tomasi_detector(img):
    corners = cv.goodFeaturesToTrack(img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    result = img_color.copy()
    kp_count = 0
    if corners is not None:
        kp_count = len(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv.circle(result, (int(x), int(y)), 5, (255, 0, 0), -1)
    return result, kp_count





def fast_detector(img):
    fast = cv.FastFeatureDetector_create(threshold=30)
    keypoints = fast.detect(img, None)
    result = cv.drawKeypoints(img_color, keypoints, None, color=(0, 255, 0))
    return result, len(keypoints)



# Feature Descriptor

def orb_detector_descriptor(img):
    orb = cv.ORB_create(nfeatures=1000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    result = cv.drawKeypoints(img_color, keypoints, None, color=(0, 255, 255))
    return result, len(keypoints), descriptors





def sift_detector_descriptor(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    result = cv.drawKeypoints(img_color, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result, len(keypoints), descriptors



img_color = cv.imread('/home/hp/Training/Opencv/week 2/detected_plate_result.jpg')

img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

harris_img, h_count = harris_corner_detector(img_gray)
shitomasi_img, st_count = shi_tomasi_detector(img_gray)
fast_img, f_count = fast_detector(img_gray)
orb_img, orb_count, orb_desc = orb_detector_descriptor(img_gray)
sift_img, sift_count, sift_desc = sift_detector_descriptor(img_gray)

show_image('Original Image', img_color)
show_image('Harris Corners (Red)', harris_img)
show_image('Shi-Tomasi (Good Features - Blue)', shitomasi_img)
show_image('FAST Keypoints (Green)', fast_img)
show_image('ORB Keypoints + Descriptors (Cyan)', orb_img)
show_image('SIFT Keypoints + Rich Descriptors', sift_img)

print(f"Harris Corners Detected       : {h_count}")
print(f"Shi-Tomasi Corners Detected   : {st_count}")
print(f"FAST Keypoints Detected       : {f_count}")
print(f"ORB Keypoints Detected        : {orb_count}")
print(f"SIFT Keypoints Detected       : {sift_count}")
print(f"ORB Descriptor Shape          : {orb_desc.shape if orb_desc is not None else None}")
print(f"SIFT Descriptor Shape         : {sift_desc.shape if sift_desc is not None else None}")

print("\nDay 8 Completed - All Feature Detectors & Descriptors (Harris, Shi-Tomasi, FAST, ORB, SIFT)")