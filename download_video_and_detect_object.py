import cv2
import requests
from ultralytics import YOLO
import tempfile
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the smallest model

# URL of a random free video
video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"  # You can replace this with other free video URLs

# Function to download a video from a URL
def download_video(url):
    response = requests.get(url, stream=True)
    video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    with open(video_file.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Downloaded video saved to: {video_file.name}")
    return video_file.name

# Download and load the video
video_path = download_video(video_url)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.data.cpu().numpy()  # Bounding boxes
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box[:6]
            label = model.names[int(class_id)]
            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
cv2.destroyAllWindows()

# Clean up the downloaded video file
os.remove(video_path)
