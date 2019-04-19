import numpy as np
import cv2
from PIL import Image

image = cv2.imread("sample/imgtilt.jpg")


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image1 = image.copy()
image = cv2.bitwise_not(image)
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image1, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

image = rotated.copy()


# image = cv2.resize(image, (1280, 720))
# image = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 40)

cv2.imshow("test", image)
cv2.waitKey(0)
