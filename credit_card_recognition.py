import cv2
import numpy as np
import json
import pytesseract


def checkSkew(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return angle


def fixSkew(image):
    angle = checkSkew(image)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return image


def processImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 59, 40)

    return image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread('sample/imgTest.jpg')
image = fixSkew(image.copy())
# image = processImage(image.copy())
image = cv2.resize(image, (1280, 720))

with open('aadhar_v1.json') as f:
    data = json.load(f)

pivotX = int(data["pivot"]["roi_min_start_x"])
pivotY = int(data["pivot"]["roi_min_start_y"])
pivotXx = int(data["pivot"]["roi_min_end_x"])
pivotYy = int(data["pivot"]["roi_min_end_y"])

cv2.rectangle(image, (pivotX, pivotY), (pivotXx, pivotYy), (0, 0, 255), 2)

for i in range(0, len(data["rois"])):
    startX = int(data["rois"][i]["roi_min_start_x"]) + pivotX
    startY = int(data["rois"][i]["roi_min_start_y"]) + pivotY
    endX = startX + int(data["rois"][i]["roi_min_width"])
    endY = startY + int(data["rois"][i]["roi_min_height"])

    roi = image[startY:endY, startX:endX]
    config = "-l eng --oem 1 --psm 11"
    text = pytesseract.image_to_string(roi, config=config, lang="ocr")
    print(data["rois"][i]["name"] + ": " + text)
    cv2.imshow("test", roi)
    cv2.waitKey(0)
