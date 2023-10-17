import cv2
import easyocr
import matplotlib.pyplot as plt

# Specify the path to the input image
image_path = r'C:\Users\isuru\OneDrive\Desktop\testimg1.png'

# Read the image using OpenCV
img = cv2.imread(image_path)

# Initialize the EasyOCR reader for English language without GPU
reader = easyocr.Reader(['en'], gpu=False)

# Perform text detection on the image
text_results = reader.readtext(img)

# Define a threshold for filtering text detection results
threshold = 0.25

# Process and display the text detection results
for bbox, text, score in text_results:
    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Display the annotated image using Matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Specify the path to the output text file
output_file = r'C:\Users\isuru\OneDrive\Desktop\text_extraction_results.txt'

# Open the output file for writing, using UTF-8 encoding
 with open(output_file, 'w', encoding='utf-8') as file:
     for result in text_results:
         text = result[1]
         file.write(text + '\n')

# Print a confirmation message indicating where the results would be saved
 print(f'Results saved to {output_file}')
