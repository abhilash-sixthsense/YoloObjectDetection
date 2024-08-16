import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(
    "yolov8m.pt"
)  # You can replace 'yolov8n.pt' with any other variant (yolov8s.pt, yolov8m.pt, etc.)

# Initialize video capture
video_path = "low_res.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the codec and create VideoWriter object if you want to save the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "output_video.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)))
)

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
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the resulting frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Write the frame to the output file
    out.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
