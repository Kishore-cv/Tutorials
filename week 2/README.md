# Training Documentation — Week 2  
### **Color, Features & Matching (5 Days Only)**  

---

## **Nov 7 ( Day 6 ) — Color Spaces & Segmentation**
### **Learned :**

#### **HSV Color Space**
- Hue = color type  
- Saturation = purity  
- Value = brightness  
- Easy to isolate a color range using only the H-channel.

#### **Color Thresholding**
- `cv.inRange(hsv, lower, upper)`  
- Output → **binary mask (0 or 255)**  
- Inside range → 255, else 0.

#### **Color Histogram**
- `cv.calcHist()` counts pixel intensity of a specific channel.  
- Used to analyze color distribution and define thresholds.

#### **Hands-On (day06_color.py)**
- Segment objects using HSV mask  
- Plot H, S, V histograms  
- Compare output before/after smoothing  

---

## **Nov 8 ( Day 7 ) — Morphological Operations**
### **Learned :**

#### **Erosion**
- Shrinks white region  
- Removes small noise

#### **Dilation**
- Expands white region  
- Fills small gaps

#### **Opening**
- Erosion → Dilation  
- Removes tiny noise blobs

#### **Closing**
- Dilation → Erosion  
- Closes holes in objects

#### **Functions**
- `cv.erode(img, kernel)`  
- `cv.dilate(img, kernel)`  
- `cv.morphologyEx(img, cv.MORPH_OPEN / MORPH_CLOSE, kernel)`

#### **Hands-On (day07_morphology.py)**
- Clean binary masks  
- Separate touching objects  

---

## **Nov 9 ( Day 8 ) — Circle Yarn Disc Task + Basic Feature Concepts**
### **Learned :** *(Simple + Blank Style)*

#### **Circle Yarn Disc Detection**
- HSV threshold → isolate yarn color  
- `cv.inRange()` → binary segmentation  
- Template-based circle crop → extract perfect disc

#### **Feature Detector Concepts**
- Feature = unique pattern  
- Corners work best for matching  
- Used for identifying same regions across images

#### **Feature Descriptor Basics**
- Convert patch around keypoint → vector  
- Later used to match two images  

---

## **Nov 10 ( Day 9 ) — FAST, Harris, SIFT, SURF**
### **Learned :**

#### **FAST Detector**
- Check brightness of 16 pixels around center  
- If many are brighter/darker → corner  
- Very fast, used in real-time applications

#### **Harris Corner Detection**
- Uses intensity gradients  
- `cv.cornerHarris(img, blockSize, ksize, k)`  
- High response = strong corner

#### **SIFT (Scale-Invariant Feature Transform)**
- Detect keypoints in different scales  
- 128-D gradient histogram descriptor  
- Robust for texture-rich images  
- Scale + rotation invariant

#### **SURF (Speeded Up Robust Features)**
- Similar to SIFT  
- Uses box filters for fast computation  
- Good for large feature sets

#### **Hands-On (day09_features.py)**
- FAST keypoint visualization  
- Harris corner map  
- Extract SIFT & SURF features  

---

## **Nov 11 ( Day 10 ) — ORB, BRIEF, BFMatcher, FLANN, Ratio Test, Homography**
### **Learned :**

#### **ORB (Oriented FAST + Rotated BRIEF)**
- FAST → keypoints  
- BRIEF → binary descriptors  
- Adds rotation compensation  
- Very fast and lightweight

#### **BRIEF Descriptor**
- Compare pixel pairs around a keypoint  
- Creates a binary vector  
- Ideal for fast matching

#### **BFMatcher (Brute Force)**
- Compare descriptor distance directly  
- `cv.BFMatcher(cv.NORM_HAMMING)` for ORB

#### **FLANN**
- Fast approximate nearest neighbor  
- Uses KD-tree / LSH for large datasets

#### **Lowe's Ratio Test**
- Accept match if `d1 < 0.75 × d2`  
- Removes wrong matches

#### **Homography Basics**
- Maps points from image A → B  
- Used for object localization  
- `cv.findHomography(src, dst, RANSAC)`

#### **Hands-On (day10_feature_matching.py)**
- ORB + BRIEF extraction  
- BFMatcher & FLANN matching  
- Apply ratio test  
- Compute homography & draw bounding box  

---

# **Week 2 Conclusion — Color, Features & Matching**
- Learned HSV-based color segmentation + histogram plotting  
- Understood morphological operations for mask cleanup  
- Completed circle yarn disc detection (segmentation + template)  
- Studied FAST, Harris, SIFT, SURF feature detectors  
- Learned ORB, BRIEF descriptors and BF/FLANN matchers  
- Implemented ratio test + homography for object detection  
- Completed a mini logo finder using feature matching  

