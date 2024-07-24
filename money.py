import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

cap = cv2.VideoCapture(0)
cap.set(3, 680)  # Set frame width to 640 pixels
cap.set(4, 90)   # Set frame height to 360 pixels

totalMoney = 0

myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 0, 'smin': 39, 'vmin': 176, 'hmax': 179, 'smax': 255, 'vmax': 255}

def empty(a):
    pass

cv2.namedWindow("settings")
cv2.resizeWindow("settings", 640, 240)
cv2.createTrackbar("Threshold1", "settings", 25, 255, empty)
cv2.createTrackbar("Threshold2", "settings", 255, 255, empty)

def preProcessing(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)
    return imgPre

while True:
    success, img = cap.read()
    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=20)
    totalMoney = 0
    imgColor = None  # Initialize imgColor here to avoid NameError
    if conFound:
        for contour in conFound:
            peri = cv2.arcLength(contour['cnt'], True)
            approx = cv2.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if len(approx) > 6:
                area = contour['area']
                imgColor, _ = myColorFinder.update(img, hsvVals)
                if area < 2700:
                    totalMoney += 1
                elif 2700 < area < 3000:
                    totalMoney += 5
                elif 3500 <= area < 4500:
                    totalMoney += 10
                else:
                    totalMoney += 2

                print(totalMoney)

    imageStacked = cvzone.stackImages([img, imgPre, imgContours], 2, 1)
    cvzone.putTextRect(imageStacked, f'Rs{totalMoney}', (50, 50))
    if imgColor is not None:  # Add this condition to avoid displaying None
        cv2.imshow("imgColor", imgColor)
    cv2.imshow("Image", imageStacked)

    imgColor, _ = myColorFinder.update(img, hsvVals)
    cv2.waitKey(1)
