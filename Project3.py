import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the motherboard image, rotate it 90 degrees, and convert it to greyscale.
OG = cv2.imread("motherboard_image.jpeg")
image = cv2.rotate(OG, cv2.ROTATE_90_CLOCKWISE)
greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Binary Thresholding.
blur = cv2.GaussianBlur(greyscale_image,(75,75), 7)
thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV, 51, 7)
 
# Perform edge detection.
dilate = cv2.dilate(thresh1, None, iterations = 20)
contours, hierarchy = cv2.findContours(dilate, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)

# Discard other contours except the main (parent).
main_contour = max(contours, key = cv2.contourArea)

# Draw the contours on the original image, and to create the mask.
image_with_contours = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR).copy()         # Better to draw contours on the original RGB image.
mask = np.zeros_like(greyscale_image)

# Refine the mask.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 50)

cv2.drawContours(mask, [main_contour], -1, (255, 255, 255), thickness = cv2.FILLED)
cv2.drawContours(image_with_contours, [main_contour], -1, (0, 255, 0), 2)

# Extract the motherboard from the image using the generated contour.
extracted_motherboard = cv2.bitwise_and(image, image, mask = mask)

width, height = int(2900/2), int(2100/2)
extracted_motherboard = cv2.resize(extracted_motherboard, (width, height))
cv2.imwrite('ExtractedMotherboard.jpg', extracted_motherboard)

# Show the extracted motherboard from the image.
cv2.imshow('Extracted Motherboard', extracted_motherboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite

# Create figures for the formal report.
titles = ['Original Image', 'Thresholded Greyscale Image',
          'Masked Image', 'Extracted Motherboard from Image']
images = [image, thresh1, mask ,extracted_motherboard]

plt.figure(figsize = (9,7))
plt.rcParams['font.family'] = 'Times New Roman'
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i], fontsize = 16)
    plt.axis('off')
plt.tight_layout()
plt.show()
