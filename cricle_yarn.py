import cv2 as cv
import numpy as np

def differencte_circle(open_mask, img):
    contours, hier = cv.findContours(open_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    
    largest_contour = None
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        (x_center, y_center), radius = cv.minEnclosingCircle(largest_contour)
        center = (int(x_center), int(y_center))
        radius = int(radius)

        x, y, w, h = cv.boundingRect(largest_contour)
        cropped_result = out[y:y+h, x:x+w]

        dummy = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.circle(dummy, center, radius, 255, -1)
        
        out_masked_img = cv.bitwise_and(img, img, mask=dummy)

    else:
        cropped_result = out

    contours, hier = cv.findContours(open_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    out_mask = np.zeros(img.shape[:2], dtype=np.uint8) 
    
    best_inner_con = None
    max_inner_area = 500

    if hier is not None and largest_contour is not None:
        hier = hier[0]
        
        outer_index = -1
        outer_area = cv.contourArea(largest_contour)
        
        for i, con in enumerate(contours):
            if abs(cv.contourArea(con) - outer_area) < 500: 
                outer_index = i
                break

        if outer_index != -1:
            current_child_index = hier[outer_index][2] 
            
            while current_child_index != -1:
                inner_con = contours[current_child_index]
                inner_area = cv.contourArea(inner_con)
                inner_peri = cv.arcLength(inner_con, True)

                if inner_area > max_inner_area and inner_peri > 0:
                    circularity = 4 * np.pi * (inner_area / (inner_peri * inner_peri))
                    if circularity > 0.6:
                        max_inner_area = inner_area
                        best_inner_con = inner_con
                
                current_child_index = hier[current_child_index][0]

    if best_inner_con is not None:
        (x_center, y_center), radius = cv.minEnclosingCircle(best_inner_con)
        center = (int(x_center), int(y_center))
        radius = int(radius)

        cv.circle(out_mask, center, radius, 255, -1)        
        res_vis = img.copy()
        cv.circle(res_vis, center, radius, (255, 0, 255), 3) 
                
    inner_mask = out_mask
    inner_mask_inv = cv.bitwise_not(inner_mask)

    ring_result = cv.bitwise_and(out_masked_img, out_masked_img, mask=inner_mask_inv)    
    
    lower_blue = np.array([100, 140, 0])
    upper_blue = np.array([255, 255, 255])
    
    ring_result_mask = cv.inRange(ring_result, lower_blue, upper_blue)
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

    ring_result_mask = cv.morphologyEx(ring_result_mask,cv.MORPH_OPEN,kernel_open, iterations=1)
    ring_result = cv.bitwise_and(ring_result, ring_result, mask=ring_result_mask)

    ring_result = cv.cvtColor(ring_result, cv.COLOR_HSV2BGR)
    ring_result = ring_result[y:y+h, x:x+w]

    return ring_result

def cricle_detect(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    croped= hsv[100:900,400:1600] 
    
    
    lower_blue = np.array([104, 120, 30])
    upper_blue = np.array([113, 205, 70])
    
    mask = cv.inRange(croped, lower_blue, upper_blue)
        
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    open_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (23,23))
    final_mask = cv.morphologyEx(open_mask, cv.MORPH_CLOSE, kernel_close)


    return differencte_circle(final_mask, croped)







    
    
image = cv.imread("/home/hp/Downloads/usr/OneDrive_1_17-11-2025/1610_uv.png")
res = cricle_detect(image)


cv.imshow("Final Result", res)
cv.waitKey(0)
cv.destroyAllWindows()