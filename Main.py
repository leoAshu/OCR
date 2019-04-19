import cv2
import numpy as np
import json
import pytesseract


def checkSkew(copy):
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    copy = cv2.bitwise_not(copy)
    thresh = cv2.threshold(copy, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return angle


def fixSkew(copy):
    angle = checkSkew(copy)

    (h, w) = copy.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    copy = cv2.warpAffine(copy, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return copy


def processImage(copy):
    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    copy = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 59, 40)

    return copy


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread('sample/DL.jpg')
image = fixSkew(image.copy())
# image = processImage(image.copy())
image = cv2.resize(image, (1280, 720))

with open('dl_US.json') as f:
    data = json.load(f)

pivotX = int(data["pivot"]["roi_min_start_x"])
pivotY = int(data["pivot"]["roi_min_start_y"])
pivotXx = int(data["pivot"]["roi_min_end_x"])
pivotYy = int(data["pivot"]["roi_min_end_y"])

# cv2.rectangle(image, (pivotX, pivotY), (pivotXx, pivotYy), (0, 0, 255), 2)

for i in range(0, len(data["rois"])):
    startX = int(data["rois"][i]["roi_min_start_x"])
    startY = int(data["rois"][i]["roi_min_start_y"])
    endX = startX + int(data["rois"][i]["roi_min_width"])
    endY = startY + int(data["rois"][i]["roi_min_height"])

    roi = image[startY:endY, startX:endX]
    config = "-l eng --oem 1 --psm 6"
    text = pytesseract.image_to_string(roi, config=config)
    print(data["rois"][i]["name"] + ": " + text)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("test", image)
    cv2.waitKey(0)

# cv2.imshow("test", image)
# cv2.waitKey(0)
