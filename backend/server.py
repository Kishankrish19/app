from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
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
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize YOLOv8 model
model = YOLO('yolov8s.pt')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Distance estimation parameters
KNOWN_WIDTHS = {
    'car': 1.8,
    'truck': 2.5,
    'bus': 2.5,
    'motorcycle': 0.8,
    'bicycle': 0.6,
    'person': 0.5,
    'default': 1.5
}
FOCAL_LENGTH = 700

# Alert thresholds
CRITICAL_DISTANCE = 2.0
WARNING_DISTANCE = 5.0

# Class colors
CLASS_COLORS = {}

# Active IP streams tracking
active_streams = {}

def get_class_color(class_id):
    """Get consistent color for each class"""
    if class_id not in CLASS_COLORS:
        np.random.seed(class_id)
        CLASS_COLORS[class_id] = tuple(np.random.randint(50, 255, 3).tolist())
    return CLASS_COLORS[class_id]

def estimate_distance(bbox_width, object_class, focal_length=FOCAL_LENGTH):
    """Estimate distance to object using focal length approximation"""
    if bbox_width > 0:
        known_width = KNOWN_WIDTHS.get(object_class, KNOWN_WIDTHS['default'])
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
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            bbox_width = x2 - x1
            distance = estimate_distance(bbox_width, class_name)
            
            alert_level = 'safe'
            if distance:
                if distance < CRITICAL_DISTANCE:
                    alert_level = 'critical'
                elif distance < WARNING_DISTANCE:
                    alert_level = 'warning'
            
            if alert_level == 'critical':
                color = (0, 0, 255)
                thickness = 3
            elif alert_level == 'warning':
                color = (0, 165, 255)
                thickness = 2
            else:
                color = get_class_color(class_id)
                thickness = 2
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{class_name} {confidence:.2f}"
            if distance:
                label += f" | {distance}m"
                if alert_level == 'critical':
                    label += " [!]"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(annotated_image, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "distance": distance,
                "bbox": [x1, y1, x2, y2],
                "alert": distance and distance < CRITICAL_DISTANCE,
                "alert_level": alert_level
            })
    
    return annotated_image, detections

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

class MobileCameraConfig(BaseModel):
    stream_url: str
    session_id: str

@api_router.get("/")
async def root():
    return {"message": "YOLOv8 Driver Assistance API", "version": "2.1 - Mobile Stream"}

@api_router.post("/camera/mobile/connect")
async def connect_mobile_camera(config: MobileCameraConfig):
    """Test connection to mobile camera stream"""
    try:
        stream_url = config.stream_url.strip()
        
        # Validate URL format
        if not (stream_url.startswith('http://') or 
                stream_url.startswith('https://') or 
                stream_url.startswith('rtsp://')):
            raise HTTPException(status_code=400, detail="Invalid stream URL. Must start with http://, https://, or rtsp://")
        
        # Try to connect to the stream
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to connect to mobile camera stream")
        
        # Read a test frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise HTTPException(status_code=400, detail="Connected but unable to read frames from stream")
        
        # Get stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        cap.release()
        
        # Store stream info
        active_streams[config.session_id] = {
            'url': stream_url,
            'connected': True,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "status": "connected",
            "message": "📱 Successfully connected to mobile camera",
            "stream_info": {
                "width": width,
                "height": height,
                "fps": fps if fps > 0 else "unknown",
                "url": stream_url
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error connecting to mobile camera: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")

@api_router.post("/camera/mobile/disconnect")
async def disconnect_mobile_camera(session_id: str):
    """Disconnect mobile camera stream"""
    if session_id in active_streams:
        del active_streams[session_id]
    return {"status": "disconnected", "message": "Mobile camera disconnected"}

@api_router.post("/detect/stream")
async def detect_from_stream(
    stream_url: str,
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Detect objects from IP camera stream (single frame)"""
    cap = None
    try:
        # Connect to stream
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to connect to stream")
        
        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Unable to read frame from stream")
        
        # Run detection
        results = model(frame, conf=0.25, iou=0.45)
        annotated_image, detections = draw_detections(frame, results)
        
        # Check for critical alerts
        critical_alerts = [d for d in detections if d['alert']]
        alert_triggered = len(critical_alerts) > 0
        
        # Log critical alerts
        if alert_triggered:
            for alert_detection in critical_alerts:
                alert_entry = AlertLog(
                    object_class=alert_detection['class'],
                    distance=alert_detection['distance'],
                    gps_latitude=gps_lat,
                    gps_longitude=gps_lon,
                    message=f"CRITICAL: {alert_detection['class']} at {alert_detection['distance']}m"
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
            "alerts": critical_alerts
        }
    
    except Exception as e:
        logging.error(f"Error in stream detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cap is not None:
            cap.release()

@api_router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Detect objects in uploaded image"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        results = model(image, conf=0.25, iou=0.45)
        annotated_image, detections = draw_detections(image, results)
        alert_triggered = any(d['alert'] for d in detections)
        
        log_entry = DetectionLog(
            detections=detections,
            gps_latitude=gps_lat,
            gps_longitude=gps_lon,
            alert_triggered=alert_triggered
        )
        doc = log_entry.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.detection_logs.insert_one(doc)
        
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
    """Process uploaded video"""
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, 'input_video.mp4')
        contents = await file.read()
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = os.path.join(temp_dir, 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=0.25, iou=0.45)
            annotated_frame, _ = draw_detections(frame, results)
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Failed to create output video")
        
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return StreamingResponse(
            BytesIO(video_data),
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=annotated_video.mp4"}
        )
    except Exception as e:
        logging.error(f"Error in video detection: {str(e)}")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    gps_lat: Optional[float] = None,
    gps_lon: Optional[float] = None
):
    """Detect objects in a single frame"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        results = model(image, conf=0.25, iou=0.45)
        annotated_image, detections = draw_detections(image, results)
        critical_alerts = [d for d in detections if d['alert']]
        alert_triggered = len(critical_alerts) > 0
        
        if alert_triggered:
            for alert_detection in critical_alerts:
                alert_entry = AlertLog(
                    object_class=alert_detection['class'],
                    distance=alert_detection['distance'],
                    gps_latitude=gps_lat,
                    gps_longitude=gps_lon,
                    message=f"CRITICAL: {alert_detection['class']} at {alert_detection['distance']}m"
                )
                doc = alert_entry.model_dump()
                doc['timestamp'] = doc['timestamp'].isoformat()
                await db.alerts.insert_one(doc)
        
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "detections": detections,
            "alert_triggered": alert_triggered,
            "alerts": critical_alerts
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

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()