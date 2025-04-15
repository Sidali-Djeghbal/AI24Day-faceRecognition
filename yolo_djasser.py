import cv2
import numpy as np
import time
from ultralytics import YOLO

def main():
    # Load the YOLOv8 face detection model
    model = YOLO('yolov8n-face.pt')  # Load a pre-trained YOLOv8 face detection model
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    # Face detection model doesn't need class filtering
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Perform detection
        start_time = time.time()
        results = model(frame)
        end_time = time.time()
        
        # Process results
        result = results[0]
        detection_frame = frame.copy()
        
        # Count faces (actually persons for YOLO's default model)
        face_count = 0
        
        # Draw bounding boxes and labels
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            
            face_count += 1
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].item()
            
            # Draw bounding box
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Face: {confidence:.2f}"
            cv2.putText(detection_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate FPS
        fps = 1.0 / (end_time - start_time)
        
        # Add FPS and detection count to the frame
        cv2.putText(detection_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(detection_frame, f"Faces detected: {face_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('YOLO Person Detection', detection_frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()