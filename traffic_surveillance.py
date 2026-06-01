# ================================
# IMPORT REQUIRED LIBRARIES
# ================================
import cv2                      # For video processing (OpenCV)
import numpy as np              # For numerical operations
from ultralytics import YOLO    # YOLO model for object detection


class TrafficMonitor:
    """
    Intelligent Traffic Monitoring System
    - Detects vehicles using YOLOv8
    - Tracks them using ByteTrack
    - Classifies into categories
    - Counts vehicles using line crossing
    """

    def __init__(self, video_path):
        """
        Constructor: Initializes model, ROI, line and counters
        """
        self.video_path = video_path

        # Load YOLO model (pre-trained on COCO dataset)
        self.model = YOLO("yolov8s.pt")

        # -------------------------------
        # ROI (Region of Interest)
        # Only vehicles inside this region will be processed
        # -------------------------------
        self.x1_roi, self.y1_roi = 200, 200   # Top-left corner
        self.x2_roi, self.y2_roi = 1500, 650  # Bottom-right corner

        # -------------------------------
        # Counting Line
        # Vehicles will be counted when crossing this line
        # -------------------------------
        self.line_y = 600
        self.offset = 15   # tolerance range for line crossing

        # -------------------------------
        # Counters for categories
        # -------------------------------
        self.count = {
            "Public Vehicle": 0,
            "Two Wheeler": 0,
            "Heavy Vehicle": 0
        }

        # To avoid counting same vehicle multiple times
        self.counted_ids = set()

    def classify_vehicle(self, class_name, x1, y1, x2, y2):
        """
        Classifies detected vehicles into:
        - Public Vehicle (cars, small trucks)
        - Two Wheeler (motorcycles)
        - Heavy Vehicle (big trucks, buses)
        """

        # Calculate bounding box area
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Two Wheeler
        if class_name == "motorcycle":
            return "Two Wheeler"

        # Trucks and buses → based on size
        if class_name in ["bus", "truck"]:
            return "Heavy Vehicle" if area > 50000 else "Public Vehicle"

        # Cars
        if class_name == "car":
            return "Public Vehicle"

        return None  # Ignore other objects

    def process_frame(self, frame):
        """
        Process each frame:
        - Detect vehicles
        - Track them
        - Filter using ROI
        - Classify
        - Count vehicles
        """

        # YOLO detection + ByteTrack tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")

        for result in results:
            # Skip if no detections
            if result.boxes is None or result.boxes.id is None:
                continue

            # Extract bounding boxes, class IDs, and tracking IDs
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy()

            for box, cls_id, track_id in zip(boxes, class_ids, track_ids):

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box)

                # Convert class ID to class name
                class_name = self.model.names[int(cls_id)]

                # Only consider relevant vehicle classes
                if class_name not in ["car", "truck", "bus", "motorcycle"]:
                    continue

                # -------------------------------
                # Compute center of bounding box
                # -------------------------------
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # -------------------------------
                # ROI FILTERING
                # Ignore vehicles outside ROI
                # -------------------------------
                if not (self.x1_roi <= cx <= self.x2_roi and
                        self.y1_roi <= cy <= self.y2_roi):
                    continue

                # -------------------------------
                # CLASSIFICATION
                # -------------------------------
                category = self.classify_vehicle(class_name, x1, y1, x2, y2)
                if category is None:
                    continue

                # -------------------------------
                # COUNTING LOGIC (IMPORTANT)
                # Count only when:
                # 1. Vehicle crosses the line
                # 2. It has not been counted before
                # 3. It is coming from above (direction control)
                # -------------------------------
                if (self.line_y - self.offset < cy < self.line_y + self.offset):
                    if int(track_id) not in self.counted_ids and cy < self.line_y:
                        self.counted_ids.add(int(track_id))
                        self.count[category] += 1

                # -------------------------------
                # DRAWING BOUNDING BOX + LABEL
                # -------------------------------
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame,
                            f"{category} ID:{int(track_id)}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        return frame

    def draw_overlay(self, frame):
        """
        Draw visual elements:
        - ROI rectangle
        - Counting line
        - Category counts
        """

        # Draw ROI
        cv2.rectangle(frame,
                      (self.x1_roi, self.y1_roi),
                      (self.x2_roi, self.y2_roi),
                      (255, 0, 0), 2)

        # Draw counting line
        cv2.line(frame,
                 (0, self.line_y),
                 (frame.shape[1], self.line_y),
                 (0, 0, 255), 3)

        # Display counts on top-left
        y_offset = 30
        for key, value in self.count.items():
            cv2.putText(frame,
                        f"{key}: {value}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)
            y_offset += 30

        return frame

    def run(self):
        """
        Main execution function:
        - Reads video
        - Processes frame-by-frame
        - Saves output video
        """

        cap = cv2.VideoCapture(self.video_path)

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            'output.mp4',
            fourcc,
            20.0,
            (int(cap.get(3)), int(cap.get(4)))
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame = self.process_frame(frame)

            # Draw overlays
            frame = self.draw_overlay(frame)

            # Save frame
            out.write(frame)

        cap.release()
        out.release()

        print("✅ Processing complete! Output saved as output.mp4")


# ================================
# RUN THE SYSTEM
# ================================
monitor = TrafficMonitor("/content/video ACV.mp4")
monitor.run()

# ================================
# DOWNLOAD OUTPUT (COLAB)
# ================================
from google.colab import files
files.download('output.mp4')
