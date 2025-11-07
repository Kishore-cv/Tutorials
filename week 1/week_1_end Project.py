import cv2 as cv
import numpy as np

class ShapeDetector:
    def __init__(self, cam_index=0, min_area=400):
        self.cap = cv.VideoCapture(cam_index)
        self.min_area = min_area
        if not self.cap.isOpened():
            print("Error: Camera not accessible.")

    def detect_shape(self, con, area, peri):
        epsilon = 0.04 * peri
        approx = cv.approxPolyDP(con, epsilon, True)
        vertices = len(approx)
        x, y, w, h = cv.boundingRect(con)
        circular = 0
        ratio = 0
        
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            aspect_ratio = w / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        else:
            if peri > 0:
                circular = 4 * np.pi * (area / (peri * peri))
            
            if circular >= 0.88:
                shape = "Circle"
            else:
                if len(con) >= 5:
                    (dx, dy), (Ma, mi), ang = cv.fitEllipse(con)
                    if min(Ma, mi) > 0:
                        ratio = max(Ma, mi) / min(Ma, mi)
                    shape = "Oval" if ratio >= 1.15 else f"Polygon-{vertices}"
                else:
                    shape = f"Polygon-{vertices}"

        M = cv.moments(con)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x + w // 2
            cy = y + h // 2

        return shape, cx, cy

    def process_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3, 3), 0.2)
        can = cv.Canny(blur, 100, 200, None, 3, False)
        
        cont, hier = cv.findContours(can, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            hier = hier[0]

        img = frame.copy()
        count = 1

        for i, con in enumerate(cont):
            area = cv.contourArea(con)
            if area < self.min_area:
                continue

            peri = cv.arcLength(con, True)
            shape, cx, cy = self.detect_shape(con, area, peri)

            cv.drawContours(img, [con], -1, (0, 255, 0), 2)

            h_next, h_prev, h_child, h_parent = hier[i]
            hierarchy_info = f"H: P={h_parent} C={h_child}"

            label = f"{shape}"
            cv.putText(img, label, (cx - 40, cy - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 0, 0), 2, cv.LINE_AA)
            cv.putText(img, hierarchy_info, (cx - 40, cy + 15), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 255), 1, cv.LINE_AA)
            
            count += 1

        return img, can

    def run(self):
        print("Live Shape Detection Started (press 'q' to quit)...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            img, can = self.process_frame(frame)
            
            cv.imshow('Live Shape Detection', img)
            cv.imshow('Canny Edges', can)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()
        print("Detection Ended.")


if __name__ == "__main__":
    detector = ShapeDetector(cam_index=0, min_area=400)
    detector.run()
