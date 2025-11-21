import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, img):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()



def bf_match(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def flann_match(des1, des2):
    index_params = dict(algorithm=1, trees=3)  
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches



img1 = cv.imread('/home/hp/Training/Opencv/Day 13/burj1.jpg')
img2 = cv.imread('/home/hp/Training/Opencv/Day 13/burj3.jpg')

if img1 is None or img2 is None:
    raise FileNotFoundError("Please place 'box.jpg' and 'box_in_scene.jpg' in the folder")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)


good_bf = bf_match(des1, des2)
good_flann = flann_match(des1.astype(np.float32), des2.astype(np.float32))

result_bf = cv.drawMatches(img1, kp1, img2, kp2, good_bf, None, 
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

result_flann = cv.drawMatches(img1, kp1, img2, kp2, good_flann, None, 
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

show_image('Original Box', img1)
show_image('Scene with Box', img2)
show_image(f'BFMatcher + Lowe Ratio - {len(good_bf)} Good Matches', result_bf)
show_image(f'FLANN + Lowe Ratio - {len(good_flann)} Good Matches', result_flann)

if len(good_bf) >= 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_bf]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_bf]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = img1.shape[:2]
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners, H)
    final = img2.copy()
    cv.polylines(final, [np.int32(warped_corners)], True, (0,255,0), 3, cv.LINE_AA)
    show_image('Object Found with Homography (BFMatcher)', cv.cvtColor(final, cv.COLOR_BGR2RGB))

print(f"BFMatcher good matches: {len(good_bf)}")
print(f"FLANN good matches: {len(good_flann)}")
print("Day 9 Completed - Feature Matching with BFMatcher & FLANN + Lowe's Ratio + Homography")