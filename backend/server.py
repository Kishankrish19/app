from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import base64
import tempfile
import shutil

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize YOLOv8 model
model = YOLO('yolov8s.pt')  # Will auto-download on first use

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Distance estimation parameters (approximate)
KNOWN_WIDTH = 1.8  # Average car width in meters
FOCAL_LENGTH = 600  # Approximate focal length (can be calibrated)

# Class colors for consistent visualization
CLASS_COLORS = {}

def get_class_color(class_id):
    """Get consistent color for each class"""
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)
        CLASS_COLORS[class_id] = tuple(np.random.randint(50, 255, 3).tolist())
    return CLASS_COLORS[class_id]

def estimate_distance(bbox_width, known_width=KNOWN_WIDTH, focal_length=FOCAL_LENGTH):
    """Estimate distance to object using focal length approximation"""
    if bbox_width > 0:
        distance = (known_width * focal_length) / bbox_width
        return round(distance, 1)
    return None

def draw_detections(image, results):
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Calculate distance
            bbox_width = x2 - x1
            distance = estimate_distance(bbox_width)
            
            # Get color for this class
            color = get_class_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Create label with distance
            label = f"{class_name} {confidence:.2f}"
            if distance:
                label += f" | {distance}m"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store detection data
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "distance": distance,
                "bbox": [x1, y1, x2, y2],
                "alert": distance and distance < 5.0
            })
    
    return annotated_image, detections

# Define Models
class DetectionLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    detections: List[dict]
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    alert_triggered: bool = False

class AlertLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    object_class: str
    distance: float
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    message: str

@api_router.get("/")
async def root():
    return {"message": "YOLOv8 Driver Assistance API"}

@api_router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Detect objects in uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(image, conf=0.3)
        
        # Draw detections
        annotated_image, detections = draw_detections(image, results)
        
        # Check for alerts
        alert_triggered = any(d['alert'] for d in detections)
        
        # Log detection
        log_entry = DetectionLog(
            detections=detections,
            gps_latitude=gps_lat,
            gps_longitude=gps_lon,
            alert_triggered=alert_triggered
        )
        doc = log_entry.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.detection_logs.insert_one(doc)
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "alert_triggered": alert_triggered
        }
    
    except Exception as e:
        logging.error(f"Error in image detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Process uploaded video and return annotated video"""
    input_path = None
    output_path = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded video to temp file
        input_path = os.path.join(temp_dir, 'input_video.mp4')
        contents = await file.read()
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default to 30 fps if unable to detect
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        output_path = os.path.join(temp_dir, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection every frame
            results = model(frame, conf=0.3)
            annotated_frame, detections = draw_detections(frame, results)
            
            # Write frame
            out.write(annotated_frame)
            
            # Store detections from first frame for summary
            if frame_count == 0:
                all_detections = detections
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Check if output file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to create output video")
        
        # Read output video
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Return video as streaming response
        return StreamingResponse(
            BytesIO(video_data),
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=annotated_video.mp4"}
        )
    
    except Exception as e:
        logging.error(f"Error in video detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Detect objects in a single frame (for live camera)"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Run detection
        results = model(image, conf=0.3)
        
        # Draw detections
        annotated_image, detections = draw_detections(image, results)
        
        # Check for alerts
        alert_triggered = any(d['alert'] for d in detections)
        alerts = [d for d in detections if d['alert']]
        
        # Log alert if triggered
        if alert_triggered and alerts:
            for alert_detection in alerts:
                alert_entry = AlertLog(
                    object_class=alert_detection['class'],
                    distance=alert_detection['distance'],
                    gps_latitude=gps_lat,
                    gps_longitude=gps_lon,
                    message=f"Warning! {alert_detection['class']} at {alert_detection['distance']}m"
                )
                doc = alert_entry.model_dump()
                doc['timestamp'] = doc['timestamp'].isoformat()
                await db.alerts.insert_one(doc)
        
        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "alert_triggered": alert_triggered,
            "alerts": alerts
        }
    
    except Exception as e:
        logging.error(f"Error in frame detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    alerts = await db.alerts.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    
    for alert in alerts:
        if isinstance(alert['timestamp'], str):
            alert['timestamp'] = datetime.fromisoformat(alert['timestamp'])
    
    return alerts

@api_router.get("/logs")
async def get_logs(limit: int = 20):
    """Get recent detection logs"""
    logs = await db.detection_logs.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit).to_list(limit)
    
    for log in logs:
        if isinstance(log['timestamp'], str):
            log['timestamp'] = datetime.fromisoformat(log['timestamp'])
    
    return logs

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()