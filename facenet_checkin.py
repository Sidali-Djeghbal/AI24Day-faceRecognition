import cv2
import numpy as np
import time
import os
import pickle
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Initialize FaceNet models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Database to store face embeddings
FACE_DB = 'face_db.pkl'

# Load or initialize face database
def load_face_db():
    if os.path.exists(FACE_DB):
        with open(FACE_DB, 'rb') as f:
            return pickle.load(f)
    return {}

face_db = load_face_db()

# Function to register new faces
def register_face(name, frame):
    try:
        # Convert frame to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(rgb_frame)
        if boxes is None or len(boxes) == 0:
            print("No faces detected")
            return False
            
        # Display faces with numbers for selection
        display_frame = frame.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, str(i+1), (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Select Face to Register', display_frame)
        cv2.waitKey(1)
        
        # Get user selection
        selected = input(f"Detected {len(boxes)} faces. Enter number (1-{len(boxes)}) to register: ")
        try:
            selected_idx = int(selected) - 1
            if selected_idx < 0 or selected_idx >= len(boxes):
                print("Invalid selection")
                return False
        except ValueError:
            print("Invalid input")
            return False
            
        # Get face embeddings in batch for efficiency
        faces = mtcnn(rgb_frame)
        if faces is None:
            print("Failed to extract faces")
            return False
            
        # Process selected face
        if faces.dim() == 5:  # [batch_size, n_faces, channels, height, width]
            faces = faces.squeeze(0)  # Remove batch dimension
        
        selected_face = faces[selected_idx].unsqueeze(0)
        embedding = resnet(selected_face).detach().numpy()[0]
        
        # Store in database
        face_db[name] = {
            'embedding': embedding,
            'checkins': []
        }
        
        # Save database
        with open(FACE_DB, 'wb') as f:
            pickle.dump(face_db, f)
        
        cv2.destroyWindow('Select Face to Register')
        return True
    except Exception as e:
        print(f"Registration error: {str(e)}")
        return False

# Function to recognize faces and log check-ins/outs
def recognize_faces(frame):
    # Convert to RGB once for efficiency
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(rgb_frame)
    
    if boxes is not None:
        # Get all face embeddings in batch for efficiency
        faces = mtcnn(rgb_frame)
        if faces is not None:
            embeddings = resnet(faces).detach().numpy()
            
            current_time = datetime.now()
            
            # Compare with known faces for each detected face
            for i, (box, embedding) in enumerate(zip(boxes, embeddings)):
                x1, y1, x2, y2 = map(int, box)
                
                # Find closest match
                min_dist = float('inf')
                match_name = None
                
                # Precompute distances for all known faces
                known_embeddings = np.array([data['embedding'] for data in face_db.values()])
                names = list(face_db.keys())
                dists = np.linalg.norm(embedding - known_embeddings, axis=1)
                
                # Find the closest match below threshold
                min_idx = np.argmin(dists)
                if dists[min_idx] < 0.8:  # Threshold for recognition
                    match_name = names[min_idx]
                
                # Draw bounding box and label for each face
                color = (0, 255, 0) if match_name else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = match_name if match_name else "Unknown"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Log check-in/out for recognized faces
                if match_name:
                    if not face_db[match_name]['checkins'] or \
                       (current_time - face_db[match_name]['checkins'][-1]['time']).total_seconds() > 60:
                        face_db[match_name]['checkins'].append({
                            'time': current_time,
                            'type': 'in'
                        })
                        print(f"{match_name} checked in at {current_time}")
            
            # Save database after processing all faces
            with open(FACE_DB, 'wb') as f:
                pickle.dump(face_db, f)
    
    return frame

def main():

    # Load YOLO for initial face detection
    model = YOLO('yolov8n-face.pt')
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("Press 'q' to quit, 'r' to register new face")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break
        
        # Perform recognition
        recognized_frame = recognize_faces(frame)
        
        # Display frame
        cv2.imshow('Face Recognition Check-In/Out', recognized_frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('r'):
            name = input("Enter name to register: ")
            if register_face(name, frame):
                print(f"{name} registered successfully!")
            else:
                print("Failed to detect face for registration")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()