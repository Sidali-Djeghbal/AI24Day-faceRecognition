# Face Detection and Recognition System

This project implements a face detection and recognition system using YOLOv8 and FAISS for efficient face detection and similarity search.

## Features

- Real-time face detection using YOLOv8
- Face recognition and similarity search using FAISS
- Face database management system
- Support for image and video processing

## Project Structure

```
├── face_database/          # Face database directory
│   ├── backups/           # Database backups
│   ├── face_database.idx  # Face database index
│   └── faiss_index.bin    # FAISS similarity search index
├── data.py                # Data processing module
├── dat.py                 # Data handling utilities
├── yolov8n-face.pt       # YOLOv8 face detection model
└── sample_images/         # Sample images for testing
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    └── 4.jpg
```

## Requirements

- Python 3.x
- PyTorch
- Ultralytics YOLOv8
- FAISS
- OpenCV

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install torch ultralytics faiss-cpu opencv-python
   ```
3. Download the YOLOv8 face detection model (already included in the repository)

## Usage

1. **Face Detection**:

   - The system uses YOLOv8 for accurate face detection in images and video streams
   - The `yolov8n-face.pt` model is optimized for face detection tasks

2. **Face Recognition**:

   - Faces are processed and stored in the FAISS database for efficient similarity search
   - The face database is maintained in the `face_database` directory

3. **Database Management**:
   - Use the data processing modules to manage the face database
   - Automatic backups are stored in the `face_database/backups` directory

## Sample Data

The repository includes sample images (1.jpg, 2.jpg, 3.jpg, 4.jpg) for testing the face detection and recognition system.

## Video Processing

The system supports video processing, as demonstrated by the included sample video file 'trying app.mp4'.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
