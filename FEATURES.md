# YOLOv8 Driver Assistance System - Enhanced Features

## 🎯 New Improvements (v2.0)

### 1. **Smart Camera Detection & Selection**
- Automatically detects all available cameras connected to your system
- Dropdown selector to choose between multiple cameras (front/rear/external)
- Real-time camera enumeration on page load
- Shows number of cameras detected

### 2. **Intelligent Proximity Alerts**
- **Critical Distance (<2m)**: RED alert with urgent voice warning
  - "Danger! [object] very close at [X] meters"
  - High-priority visual notification
  - Immediate voice alert with faster speech rate
  
- **Approaching Objects**: ORANGE warning with monitoring
  - Tracks object velocity over time
  - Detects objects moving closer to you
  - "Notice: [object] approaching at [X] meters"
  - Smart throttling to avoid alert spam

- **Nearby Objects (2-5m)**: YELLOW indicator
  - Visual monitoring without audio alerts
  - Helps maintain situational awareness

### 3. **Object Movement Tracking**
- Maintains history of detected objects across frames
- Calculates velocity to detect approaching objects
- Uses sliding window (last 5 frames) for accurate velocity calculation
- Distinguishes between static and moving objects

### 4. **Enhanced Detection System**
- Class-specific distance estimation (cars, trucks, persons have different reference widths)
- Improved focal length calibration (700px)
- Better confidence thresholds (0.25) and IoU (0.45)
- Visual feedback with color-coded bounding boxes:
  - Red: Critical (<2m)
  - Orange: Approaching
  - Color-coded by class: Safe distance

### 5. **Improved Alert Management**
- Separate alert throttling for critical vs approaching objects
- Critical alerts: Max 1 per object every 3 seconds
- Approaching alerts: Max 1 per object every 5 seconds
- Alert history with timestamps in UI

### 6. **Better UI/UX**
- Clear visual legend explaining alert thresholds
- Status badges: CRITICAL | APPROACHING | NEARBY | SAFE
- Frame counter during live detection
- Camera info display
- Improved table layout with status columns

### 7. **Performance Optimizations**
- Faster frame processing (500ms interval = 2 FPS)
- Efficient object tracking with minimal memory overhead
- Optimized voice synthesis (cancels previous alerts for urgent ones)
- Better video codec handling

## 🚀 Usage Guide

### Live Camera Mode
1. Grant camera permissions when prompted
2. If multiple cameras detected, select your preferred camera from dropdown
3. Click "Start Camera"
4. System will:
   - Process frames every 0.5 seconds
   - Track objects and calculate distances
   - Alert you only when objects are <2m (CRITICAL)
   - Notify when objects are approaching
   - Display all nearby objects (within 5m)

### Image Mode
- Upload any image
- Get instant detection with distance estimates
- See color-coded alerts for close objects

### Video Mode
- Upload video file
- System processes every frame
- Downloads annotated video with bounding boxes

## 🎨 Alert System Colors

| Color | Distance | Behavior |
|-------|----------|----------|
| 🔴 RED | <2m | CRITICAL - Urgent voice alert |
| 🟠 ORANGE | Approaching | Voice notification |
| 🟡 YELLOW | 2-5m | Visual indicator only |
| 🟢 GREEN | >5m | Safe - Normal display |

## 📊 Technical Specifications

- **Model**: YOLOv8s (21.5MB)
- **Confidence Threshold**: 25%
- **IoU Threshold**: 45%
- **Processing Speed**: ~2 FPS (live camera)
- **Distance Accuracy**: ±20% (depends on calibration)
- **Supported Objects**: 80 COCO classes
- **Voice Engine**: Web Speech API
- **GPS**: Browser Geolocation with fallback

## 🔧 Customization

You can adjust these parameters in the code:

**Backend** (`server.py`):
```python
CRITICAL_DISTANCE = 2.0  # meters - trigger alert
WARNING_DISTANCE = 5.0   # meters - monitor for approaching
FOCAL_LENGTH = 700       # calibration parameter
```

**Frontend** (`App.js`):
```javascript
CRITICAL_DISTANCE = 2.0  // meters
WARNING_DISTANCE = 5.0   // meters
detectionInterval = 500  // ms (processing frequency)
```

## 🎯 Future Enhancements

Possible additions:
- Lane detection overlay
- Traffic sign recognition
- Multi-object tracking IDs
- Distance calibration wizard
- Alert sound customization
- Dashboard recording
- Integration with vehicle CAN bus
- Night mode optimization
