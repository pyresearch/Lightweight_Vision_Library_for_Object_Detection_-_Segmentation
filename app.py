import cv2
import time
import random
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Full class list for YOLOv8 (COCO dataset classes)
classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
    "sofa", "potted plant", "bed", "dining table", "toilet", "TV monitor", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate a color map for the classes
color_map = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in classes}

# Path to the YOLOv8 model (download manually if not available)
yolov8_model_path = "yolo11x.pt"

# Load the YOLOv8 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolo11',
    model_path=yolov8_model_path,
    confidence_threshold=0.5,
    device="cuda:0"  # or 'cpu'
)

# Open video file
video_path = "demo3.mp4"
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video settings
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Total frames for progress calculation
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate and display progress
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    progress = int(frame_number / total_frames * 100)
    print(f"Processing: {progress}%")

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Sliced Inference with YOLOv8
    start_time = time.time()
    result = get_sliced_prediction(
        frame_rgb,
        detection_model,
        slice_height=height,
        slice_width=width // 2,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    time_detection = time.time() - start_time
    print(f"Detection time: {time_detection:.2f} seconds")

    # Draw bounding boxes and labels
    object_prediction_list = result.object_prediction_list
    for obj in object_prediction_list:
        bbox = obj.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        label = obj.category.name

        # Handle unknown labels (Optional, for robustness)
        if label not in color_map:
            color_map[label] = (255, 255, 255)  # Assign a default color (white)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[label], 2)

        # Draw label
        label_text = f"{label}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[label], 2)

    # Write the frame to the output video
    out.write(frame)

    # Optionally display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output saved to", output_path)
