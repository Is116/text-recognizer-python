import cv2
import easyocr
import matplotlib.pyplot as plt

# Define the image path
image_path = '/home/phillip/Desktop/todays_tutorial/30_text_detection_easyocr/code/data/test2.png'

# Read the image
img = cv2.imread(image_path)

# Create an EasyOCR Reader instance
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on the image
text_results = reader.readtext(img)

# Set the threshold for text detection
threshold = 0.25

# Draw bounding boxes and text on the image
for bbox, text, score in text_results:
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Display the image with detected text
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
