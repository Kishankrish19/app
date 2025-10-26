import { useState, useRef, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Camera, Upload, Video, AlertTriangle, MapPin } from "lucide-react";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [mode, setMode] = useState("camera");
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [gpsLocation, setGpsLocation] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [recentLogs, setRecentLogs] = useState([]);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const speechSynthesisRef = useRef(window.speechSynthesis);
  const lastAlertTimeRef = useRef(0);

  // Request GPS location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.watchPosition(
        (position) => {
          setGpsLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error) => {
          console.log("GPS not available, using simulated location");
          // Simulated GPS for demo
          setGpsLocation({
            latitude: 12.9716,
            longitude: 77.5946
          });
        }
      );
    } else {
      // Fallback to simulated GPS
      setGpsLocation({
        latitude: 12.9716,
        longitude: 77.5946
      });
    }
  }, []);

  // Fetch recent alerts
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await axios.get(`${API}/alerts`);
        setAlerts(response.data);
      } catch (error) {
        console.error("Error fetching alerts:", error);
      }
    };
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 10000); // Update every 10s
    return () => clearInterval(interval);
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 1280, height: 720 }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
        toast.success("Camera started");
        
        // Start detection loop
        startDetectionLoop();
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast.error("Failed to access camera. Please check permissions.");
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setCameraActive(false);
      
      // Stop detection loop
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      
      toast.info("Camera stopped");
    }
  };

  // Detection loop for live camera
  const startDetectionLoop = () => {
    detectionIntervalRef.current = setInterval(async () => {
      await captureAndDetect();
    }, 1000); // Process every 1 second
  };

  // Capture frame and send for detection
  const captureAndDetect = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert canvas to blob
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      
      if (gpsLocation) {
        formData.append('gps_lat', gpsLocation.latitude);
        formData.append('gps_lon', gpsLocation.longitude);
      }
      
      try {
        const response = await axios.post(`${API}/detect/frame`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        
        setResult(response.data);
        
        // Handle alerts with voice
        if (response.data.alert_triggered && response.data.alerts.length > 0) {
          const now = Date.now();
          // Throttle voice alerts to once every 3 seconds
          if (now - lastAlertTimeRef.current > 3000) {
            lastAlertTimeRef.current = now;
            const alert = response.data.alerts[0];
            speakAlert(`Warning! ${alert.class} ahead at ${alert.distance} meters`);
          }
        }
      } catch (error) {
        console.error("Error detecting frame:", error);
      }
    }, 'image/jpeg', 0.8);
  };

  // Voice alert using Web Speech API
  const speakAlert = (text) => {
    if (speechSynthesisRef.current) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.2;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      speechSynthesisRef.current.speak(utterance);
    }
  };

  // Handle image upload
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    
    if (gpsLocation) {
      formData.append('gps_lat', gpsLocation.latitude);
      formData.append('gps_lon', gpsLocation.longitude);
    }
    
    try {
      const response = await axios.post(`${API}/detect/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResult(response.data);
      toast.success("Image processed successfully");
      
      if (response.data.alert_triggered) {
        toast.error("⚠ Collision Risk Detected!");
      }
    } catch (error) {
      console.error("Error processing image:", error);
      toast.error("Failed to process image");
    } finally {
      setProcessing(false);
    }
  };

  // Handle video upload
  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    
    if (gpsLocation) {
      formData.append('gps_lat', gpsLocation.latitude);
      formData.append('gps_lon', gpsLocation.longitude);
    }
    
    try {
      const response = await axios.post(`${API}/detect/video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });
      
      // Create download link for processed video
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'annotated_video.mp4');
      document.body.appendChild(link);
      link.click();
      link.remove();
      
      toast.success("Video processed! Download started.");
      setResult({ video_processed: true });
    } catch (error) {
      console.error("Error processing video:", error);
      toast.error("Failed to process video");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <div className="header-content">
          <div className="header-title">
            <Camera className="header-icon" />
            <h1 data-testid="app-title">YOLO Drive Assist</h1>
          </div>
          {gpsLocation && (
            <div className="gps-display" data-testid="gps-location">
              <MapPin size={16} />
              <span>{gpsLocation.latitude.toFixed(4)}, {gpsLocation.longitude.toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="left-panel">
          <Card className="mode-selector-card">
            <CardHeader>
              <CardTitle>Detection Mode</CardTitle>
              <CardDescription>Choose input source for object detection</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs value={mode} onValueChange={(value) => {
                setMode(value);
                if (value !== "camera" && cameraActive) {
                  stopCamera();
                }
                setResult(null);
              }}>
                <TabsList className="mode-tabs">
                  <TabsTrigger value="camera" data-testid="mode-camera">
                    <Camera size={18} />
                    <span>Live Camera</span>
                  </TabsTrigger>
                  <TabsTrigger value="image" data-testid="mode-image">
                    <Upload size={18} />
                    <span>Image</span>
                  </TabsTrigger>
                  <TabsTrigger value="video" data-testid="mode-video">
                    <Video size={18} />
                    <span>Video</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="camera" className="mode-content">
                  <div className="camera-controls">
                    {!cameraActive ? (
                      <Button onClick={startCamera} className="start-camera-btn" data-testid="start-camera-btn">
                        <Camera size={18} />
                        Start Camera
                      </Button>
                    ) : (
                      <Button onClick={stopCamera} variant="destructive" data-testid="stop-camera-btn">
                        Stop Camera
                      </Button>
                    )}
                  </div>
                  
                  <div className="video-container">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="camera-feed"
                      data-testid="camera-video"
                    />
                    <canvas ref={canvasRef} style={{ display: 'none' }} />
                  </div>
                </TabsContent>

                <TabsContent value="image" className="mode-content">
                  <div className="upload-container">
                    <label htmlFor="image-upload" className="upload-label" data-testid="image-upload-label">
                      <Upload size={32} />
                      <span>Click to upload image</span>
                      <input
                        id="image-upload"
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="upload-input"
                        data-testid="image-upload-input"
                      />
                    </label>
                  </div>
                </TabsContent>

                <TabsContent value="video" className="mode-content">
                  <div className="upload-container">
                    <label htmlFor="video-upload" className="upload-label" data-testid="video-upload-label">
                      <Video size={32} />
                      <span>Click to upload video</span>
                      <input
                        id="video-upload"
                        type="file"
                        accept="video/*"
                        onChange={handleVideoUpload}
                        className="upload-input"
                        data-testid="video-upload-input"
                      />
                    </label>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {processing && (
            <Card className="processing-card">
              <CardContent className="processing-content">
                <div className="loading-spinner" />
                <p>Processing...</p>
              </CardContent>
            </Card>
          )}

          {result && result.image && (
            <Card className="result-card">
              <CardHeader>
                <CardTitle>Detection Result</CardTitle>
              </CardHeader>
              <CardContent>
                <img src={result.image} alt="Detected" className="result-image" data-testid="result-image" />
              </CardContent>
            </Card>
          )}
        </div>

        <div className="right-panel">
          {result && result.detections && result.detections.length > 0 && (
            <Card className="detections-card">
              <CardHeader>
                <CardTitle>Detected Objects</CardTitle>
                <CardDescription>{result.detections.length} objects detected</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="detections-table" data-testid="detections-table">
                  <div className="table-header">
                    <span>Object</span>
                    <span>Confidence</span>
                    <span>Distance</span>
                  </div>
                  {result.detections.map((detection, idx) => (
                    <div key={idx} className={`table-row ${detection.alert ? 'alert-row' : ''}`} data-testid={`detection-row-${idx}`}>
                      <span>{detection.class}</span>
                      <span>{(detection.confidence * 100).toFixed(0)}%</span>
                      <span className={detection.alert ? 'alert-distance' : ''}>
                        {detection.distance ? `${detection.distance}m` : 'N/A'}
                        {detection.alert && <AlertTriangle size={14} className="alert-icon" />}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {result && result.video_processed && (
            <Card className="video-result-card">
              <CardContent className="video-result-content">
                <Video size={48} />
                <p>Video processed successfully!</p>
                <p className="video-result-desc">Check your downloads folder</p>
              </CardContent>
            </Card>
          )}

          {alerts.length > 0 && (
            <Card className="alerts-card">
              <CardHeader>
                <CardTitle className="alerts-title">
                  <AlertTriangle size={18} />
                  Recent Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="alerts-list" data-testid="alerts-list">
                  {alerts.slice(0, 5).map((alert, idx) => (
                    <div key={alert.id || idx} className="alert-item" data-testid={`alert-item-${idx}`}>
                      <div className="alert-content">
                        <span className="alert-object">{alert.object_class}</span>
                        <span className="alert-distance">{alert.distance}m</span>
                      </div>
                      <span className="alert-time">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;