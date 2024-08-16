import cv2
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8m.pt')  # Use the smallest model

# List of URLs to download random images
image_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/2/2d/Snake_River_%285mb%29.jpg',
    # "https://source.unsplash.com/random/800x600",  # Random image from Unsplash
    # "https://picsum.photos/800/600",               # Random image from Picsum
    # "https://loremflickr.com/800/600",             # Random image from LoremFlickr
]

# Function to download an image from a URL
def download_image(url):
    print("Downloading image")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    print('Download completed')
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert PIL image to OpenCV format

for url in image_urls:
    try:
        # Download and load the image
        frame = download_image(url)

        # Run YOLOv8 inference on the image
        results = model(frame)

        # Draw bounding boxes and labels on the image
        for result in results:
            boxes = result.boxes.data.cpu().numpy()  # Bounding boxes
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box[:6]
                label = model.names[int(class_id)]
                # Draw rectangle and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('YOLOv8 Object Detection', frame)
        print("Waiting for user input")
        # To Wait for user input to close the image chang wait key to 0
        cv2.waitKey(5000)
        print("")

    except Exception as e:
        print(f"Error processing image from {url}: {e}")

# Close all OpenCV windows
cv2.destroyAllWindows()
