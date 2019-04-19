import numpy as np
import pytesseract
import cv2
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# size = 7016, 4961
# image1 = Image.open("sample/DL.jpg")
# image1 = image1.resize(size, Image.ANTIALIAS)
#
# image = np.array(image1)

image = cv2.imread('sample/DL2.jpg')

image = cv2.resize(image, (1280, 720))

# Convert RGB to BGR
# image = image[:, :, ::-1].copy()

# image = image[130:150, 200:300]

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 40)

# topLeft 1873 1366
# bottomRight 1968 1491
# topLeft 46 260
# bottomRight 141 385
# topLeft 45 259
# bottomRight 140 384
# topLeft 371 1271
# bottomRight 466 1396

cv2.rectangle(image, (780, 455), (920, 490), (0, 0, 255), 2)

# cv2.imwrite('C:\\Users\\OPTLPTP095\\Desktop\\opencv-text-recognition\\sample\\dl.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

cv2.imshow("sample_display", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
