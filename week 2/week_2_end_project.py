#################################################
######     FingerPrint Detection          #######
#################################################


import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def show_image(name, img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis('off')
    plt.show()

 
def preprocess_fingerprint(image_path):
    image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
    _, img_bin = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    print(f"Shape of image:{image.shape}\n")
    return img_bin
 
def match_fingerprints(img1, img2):
    img1 = preprocess_fingerprint(img1)
    img2 = preprocess_fingerprint(img2)
 
    sift = cv.SIFT_create(nfeatures=200)
 
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  
 
    index_params = dict(algorithm=1, trees=3)  
    search_params = dict(checks=150)  
    flann = cv.FlannBasedMatcher(index_params, search_params)
 
    matches = flann.knnMatch(des1, des2, k=2)
 
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
 
    match_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print("Matches:",good_matches)
    show_image("Output Image ",match_img)

    confidences = (len(good_matches)/min(len(kp1),len(kp2)))*100
    print("Confidences scorce :",confidences,"Match Image Length : ",len(good_matches)," kp1 length :",len(kp1)," Length of kp2 :",len(kp2))
    res = "Matched" if confidences > 50 else "Not Matched"
    return res

result = match_fingerprints("/home/hp/Training/Opencv/Day 9/fingerprint1.jpg","/home/hp/Training/Opencv/Day 9/fingerprint2.jpg")
print("Fingerprint : ",result)

