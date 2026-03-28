import cv2
from ultralytics import YOLO

class MultimodalSensor:
    def __init__(self):
        print("Booting Multimodal AI: YOLOv8 (Dashcam) + OpenCV Vision (Cabin)...")
        
        # 1. Initialize YOLO (For the square bounding boxes on objects)
        self.yolo_model = YOLO('yolov8n.pt') 
        
        # 2. Initialize OpenCV Face Detection (Rock-solid fallback for MediaPipe)
        # This uses the pre-trained AI that comes built-in with your cv2 installation
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame):
        """Processes the frame through BOTH YOLO and OpenCV simultaneously."""
        
        # --- MODEL 1: YOLO OBJECT DETECTION ---
        results = self.yolo_model(frame, stream=False, verbose=False)
        annotated_frame = results[0].plot() 
        
        detected_classes = []
        max_proximity = 0  # NEW: We will track how close the largest object is!
        frame_area = frame.shape[0] * frame.shape[1]

        for box in results[0].boxes:
            detected_classes.append(self.yolo_model.names[int(box.cls)])
            
            # CALCULATE DISTANCE: Bounding Box Area / Screen Area
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = (x2 - x1) * (y2 - y1)
            proximity_perc = (box_area / frame_area) * 100
            
            if proximity_perc > max_proximity:
                max_proximity = proximity_perc
                
        # Scale it up slightly so you don't have to touch your nose to the camera
        live_distance_metric = min(100, max_proximity * 2.5)

        # --- MODEL 2: OPENCV DRIVER FACE DETECTION ---
        gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        driver_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "Driver Tracked", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # --- LOGIC INTEGRATION ---
        scene_description = "Clear Road"
        fatigue_warning = False

        if detected_classes:
            unique_objects = set(detected_classes)
            scene_description = f"Detected: {', '.join(unique_objects)}"
            if "person" in unique_objects and not driver_detected:
                fatigue_warning = True 

        # We now return FOUR variables, including the live physical distance!
        return annotated_frame, fatigue_warning, scene_description, live_distance_metric