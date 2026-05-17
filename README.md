# AI-Powered Traffic Monitoring and Vehicle Counting System

## Project Overview

This project is a real-time traffic monitoring system that performs vehicle detection, multi-object tracking, and lane-wise vehicle counting from uploaded traffic videos.

The system uses a custom-trained YOLOv11 model for vehicle detection and DeepSORT for persistent object tracking across frames. A FastAPI backend handles video processing requests, Streamlit provides the user interface, MLflow tracks training experiments, and Docker containerizes the complete application for reproducible deployment.

The complete workflow is fully containerized and runs end-to-end using Docker Compose.

---

## Features

- Real-time vehicle detection using custom-trained YOLOv11
- Multi-object tracking using DeepSORT
- Lane/zone-based vehicle counting
- Vehicle ID persistence to avoid double counting
- FastAPI backend for video processing
- Streamlit UI for video upload and visualization
- MLflow experiment tracking
- Fully Dockerized deployment
- Modular and scalable architecture

---

## System Workflow

1. User uploads traffic video through Streamlit UI  
2. Streamlit sends video to FastAPI backend  
3. Backend processes frames using YOLOv11 detection  
4. DeepSORT assigns persistent tracking IDs  
5. Vehicles crossing the counting line/zone are counted once  
6. Processed video and statistics are returned to the UI  

---

# System Architecture Diagram



---

## Tech Stack

| Component | Technology |
|---|---|
| Detection Model | YOLOv11 |
| Tracking | DeepSORT |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Language | Python |
| Video Processing | OpenCV |

---

## Project Structure

```bash
.
├── api.py
├── app.py
├── tracker.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── models/
├── outputs/
├── videos/
└── mlruns/
```

---

## Setup & Run

### Clone Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

---

### Build and Start Containers

```bash
docker compose up --build
```

---

## Access Services

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI Backend | http://localhost:8003 |
| MLflow UI | http://localhost:5000 |

---

## Docker Services

| Service | Purpose |
|---|---|
| streamlit | Frontend user interface |
| api | Video processing backend |
| mlflow | Experiment tracking server |

All services share a single Docker image for faster startup and reduced build time.

---

## MLflow Experiment Tracking

MLflow is used to track:

- Training metrics
- Precision and recall
- mAP scores
- Hyperparameters
- Model artifacts
- Training visualizations

Tracked experiments can be viewed through the MLflow dashboard.

---

## Vehicle Counting Logic

- YOLOv11 detects vehicles in each frame
- DeepSORT assigns unique tracking IDs
- A counting line/zone is defined
- Each vehicle is counted only once when crossing the zone
- Persistent IDs help prevent double counting across frames

---

## Future Improvements

- Live CCTV/RTSP stream support
- Multi-camera distributed deployment
- Cloud deployment on AWS/GCP
- Database integration for analytics
- Automatic traffic congestion analysis
- Kubernetes-based scaling

---

## Demo Flow

1. Upload traffic video  
2. Run vehicle detection and tracking  
3. Visualize processed video  
4. View total vehicle count  
5. Inspect MLflow experiment logs  

---

## Authors

- Khushi B