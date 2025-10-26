import { useState, useRef, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Camera, Upload, Video, AlertTriangle, MapPin, TrendingDown, Bell, Smartphone, Wifi } from "lucide-react";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CRITICAL_DISTANCE = 2.0;
const WARNING_DISTANCE = 5.0;

function App() {
  const [mode, setMode] = useState("camera");
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [gpsLocation, setGpsLocation] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  const [cameraSource, setCameraSource] = useState("local"); // 'local' or 'mobile'
  const [mobileStreamUrl, setMobileStreamUrl] = useState("");
  const [mobileConnected, setMobileConnected] = useState(false);
  const [objectHistory, setObjectHistory] = useState({});
  const [sessionId] = useState(() => Math.random().toString(36).substring(7));
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const speechSynthesisRef = useRef(window.speechSynthesis);
  const lastAlertTimeRef = useRef({});
  const frameCountRef = useRef(0);
  const mobileStreamRef = useRef(null);

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
          setGpsLocation({ latitude: 12.9716, longitude: 77.5946 });
        }
      );
    } else {
      setGpsLocation({ latitude: 12.9716, longitude: 77.5946 });
    }
  }, []);

  // Enumerate available cameras
  useEffect(() => {
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (error) {
        console.error("Error enumerating cameras:", error);
      }
    };
    getCameras();
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
    const interval = setInterval(fetchAlerts, 10000);
    return () => clearInterval(interval);
  }, []);

  // Connect to mobile camera
  const connectMobileCamera = async () => {
    if (!mobileStreamUrl.trim()) {
      toast.error("Please enter a valid stream URL");
      return;
    }

    try {
      setProcessing(true);
      const response = await axios.post(`${API}/camera/mobile/connect`, {
        stream_url: mobileStreamUrl,
        session_id: sessionId
      });

      setMobileConnected(true);
      toast.success(response.data.message);
      
      // Auto-start camera after successful connection
      setTimeout(() => {
        startMobileCamera();
      }, 500);
    } catch (error) {
      console.error("Mobile connection error:", error);
      toast.error(error.response?.data?.detail || "Failed to connect to mobile camera");
      setMobileConnected(false);
    } finally {
      setProcessing(false);
    }
  };

  // Start mobile camera stream
  const startMobileCamera = () => {
    setCameraActive(true);
    frameCountRef.current = 0;
    setObjectHistory({});
    startMobileDetectionLoop();
  };

  // Mobile detection loop
  const startMobileDetectionLoop = () => {
    detectionIntervalRef.current = setInterval(async () => {
      await captureMobileFrame();
    }, 500);
  };

  // Capture frame from mobile stream
  const captureMobileFrame = async () => {
    try {
      frameCountRef.current++;
      
      const response = await axios.post(
        `${API}/detect/stream`,
        null,
        {
          params: {
            stream_url: mobileStreamUrl,
            gps_lat: gpsLocation?.latitude,
            gps_lon: gpsLocation?.longitude
          }
        }
      );

      // Analyze detections for approaching objects
      const enrichedDetections = response.data.detections.map(detection => {
        const movement = analyzeObjectMovement(
          detection.class,
          detection.distance || 10,
          detection.bbox
        );
        return {
          ...detection,
          isApproaching: movement.isApproaching,
          velocity: movement.velocity
        };
      });

      setResult({
        ...response.data,
        detections: enrichedDetections
      });

      // Handle alerts
      handleAlerts(enrichedDetections);
    } catch (error) {
      console.error("Mobile frame capture error:", error);
      // Auto-fallback to local camera
      if (cameraActive && mobileConnected) {
        toast.error("Lost mobile connection. Attempting fallback to local camera...");
        await fallbackToLocalCamera();
      }
    }
  };

  // Fallback to local camera
  const fallbackToLocalCamera = async () => {
    stopCamera();
    setMobileConnected(false);
    setCameraSource("local");
    toast.info("Switched to local camera");
    
    // Auto-start local camera
    setTimeout(() => {
      startCamera();
    }, 1000);
  };

  // Start local camera
  const startCamera = async () => {
    try {
      const constraints = {
        video: {
          deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
        frameCountRef.current = 0;
        setObjectHistory({});
        toast.success("Local camera started");
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
    }
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    setCameraActive(false);
    setObjectHistory({});
    frameCountRef.current = 0;
    
    if (mobileConnected) {
      axios.post(`${API}/camera/mobile/disconnect`, null, {
        params: { session_id: sessionId }
      }).catch(console.error);
      setMobileConnected(false);
    }
    
    toast.info("Camera stopped");
  };

  // Local camera detection loop
  const startDetectionLoop = () => {
    detectionIntervalRef.current = setInterval(async () => {
      await captureAndDetect();
    }, 500);
  };

  // Analyze object movement
  const analyzeObjectMovement = (objectClass, currentDistance, bbox) => {
    const objectKey = objectClass;
    const now = Date.now();
    
    if (!objectHistory[objectKey]) {
      objectHistory[objectKey] = {
        distances: [currentDistance],
        timestamps: [now],
        lastAlertTime: 0
      };
      return { isApproaching: false, velocity: 0 };
    }
    
    const history = objectHistory[objectKey];
    history.distances.push(currentDistance);
    history.timestamps.push(now);
    
    if (history.distances.length > 5) {
      history.distances.shift();
      history.timestamps.shift();
    }
    
    if (history.distances.length >= 3) {
      const oldDistance = history.distances[0];
      const timeDiff = (now - history.timestamps[0]) / 1000;
      const velocity = (currentDistance - oldDistance) / timeDiff;
      const isApproaching = velocity < -0.3 && currentDistance < WARNING_DISTANCE;
      return { isApproaching, velocity: Math.abs(velocity) };
    }
    
    return { isApproaching: false, velocity: 0 };
  };

  // Handle alerts
  const handleAlerts = (detections) => {
    const now = Date.now();
    detections.forEach(detection => {
      const objectKey = detection.class;
      
      if (detection.distance && detection.distance < CRITICAL_DISTANCE) {
        if (!lastAlertTimeRef.current[objectKey] || 
            now - lastAlertTimeRef.current[objectKey] > 3000) {
          lastAlertTimeRef.current[objectKey] = now;
          speakAlert(`Danger! ${detection.class} very close at ${detection.distance} meters`, true);
          toast.error(`⚠️ CRITICAL: ${detection.class} at ${detection.distance}m`, { duration: 3000 });
        }
      }
      else if (detection.isApproaching && detection.distance < WARNING_DISTANCE) {
        if (!lastAlertTimeRef.current[`${objectKey}_approach`] || 
            now - lastAlertTimeRef.current[`${objectKey}_approach`] > 5000) {
          lastAlertTimeRef.current[`${objectKey}_approach`] = now;
          speakAlert(`Notice: ${detection.class} approaching at ${detection.distance} meters`, false);
          toast.warning(`⚡ ${detection.class} approaching - ${detection.distance}m`, { duration: 2000 });
        }
      }
    });
  };

  // Capture and detect from local camera
  const captureAndDetect = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    frameCountRef.current++;
    
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
        
        const enrichedDetections = response.data.detections.map(detection => {
          const movement = analyzeObjectMovement(
            detection.class,
            detection.distance || 10,
            detection.bbox
          );
          return {
            ...detection,
            isApproaching: movement.isApproaching,
            velocity: movement.velocity
          };
        });
        
        setResult({
          ...response.data,
          detections: enrichedDetections
        });
        
        handleAlerts(enrichedDetections);
      } catch (error) {
        console.error("Error detecting frame:", error);
      }
    }, 'image/jpeg', 0.8);
  };

  // Voice alert
  const speakAlert = (text, urgent = false) => {
    if (speechSynthesisRef.current) {
      if (urgent) {
        speechSynthesisRef.current.cancel();
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = urgent ? 1.3 : 1.0;
      utterance.pitch = urgent ? 1.2 : 1.0;
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
        toast.error("⚠ Objects detected within critical distance!");
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

  // Get status color
  const getStatusColor = (detection) => {
    if (detection.distance && detection.distance < CRITICAL_DISTANCE) {
      return 'critical';
    } else if (detection.isApproaching) {
      return 'approaching';
    } else if (detection.distance && detection.distance < WARNING_DISTANCE) {
      return 'warning';
    }
    return 'safe';
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <div className="header-content">
          <div className="header-title">
            <Camera className="header-icon" />
            <h1 data-testid="app-title">YOLO Drive Assist</h1>
            {mobileConnected && (
              <div className="mobile-badge">
                <Smartphone size={16} />
                <span>Mobile Connected</span>
              </div>
            )}
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
                  {/* Camera Source Selector */}
                  <div className="source-selector" data-testid="source-selector">
                    <Label>Camera Source:</Label>
                    <Select value={cameraSource} onValueChange={setCameraSource} disabled={cameraActive}>
                      <SelectTrigger className="source-select-trigger">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local">Local PC/Laptop Camera</SelectItem>
                        <SelectItem value="mobile">📱 Mobile Camera (IP Stream)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {cameraSource === "local" && (
                    <>
                      {availableCameras.length > 1 && (
                        <div className="camera-selector" data-testid="camera-selector">
                          <Label>Select Camera:</Label>
                          <Select value={selectedCamera} onValueChange={setSelectedCamera} disabled={cameraActive}>
                            <SelectTrigger className="camera-select-trigger">
                              <SelectValue placeholder="Choose camera" />
                            </SelectTrigger>
                            <SelectContent>
                              {availableCameras.map((camera, idx) => (
                                <SelectItem key={camera.deviceId} value={camera.deviceId}>
                                  {camera.label || `Camera ${idx + 1}`}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                      
                      <div className="camera-info" data-testid="camera-info">
                        <p className="info-text">📹 {availableCameras.length} camera(s) detected</p>
                      </div>
                      
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
                    </>
                  )}

                  {cameraSource === "mobile" && (
                    <>
                      <div className="mobile-stream-config">
                        <Label htmlFor="stream-url">Mobile Camera Stream URL</Label>
                        <Input
                          id="stream-url"
                          type="text"
                          placeholder="http://192.168.1.100:8080/video"
                          value={mobileStreamUrl}
                          onChange={(e) => setMobileStreamUrl(e.target.value)}
                          disabled={cameraActive}
                          className="stream-url-input"
                          data-testid="mobile-stream-url"
                        />
                        <div className="mobile-help">
                          <p>📱 <strong>Setup Instructions:</strong></p>
                          <ol>
                            <li>Install "IP Webcam" or "DroidCam" on your phone</li>
                            <li>Start the camera server in the app</li>
                            <li>Copy the stream URL shown (e.g., http://192.168.1.100:8080/video)</li>
                            <li>Paste it above and click Connect</li>
                          </ol>
                          <p className="mobile-note">⚠️ Ensure your phone and PC are on the same network</p>
                        </div>
                      </div>
                      
                      <div className="camera-controls">
                        {!mobileConnected ? (
                          <Button 
                            onClick={connectMobileCamera} 
                            className="connect-mobile-btn" 
                            disabled={processing}
                            data-testid="connect-mobile-btn"
                          >
                            <Wifi size={18} />
                            {processing ? "Connecting..." : "Connect Mobile Camera"}
                          </Button>
                        ) : !cameraActive ? (
                          <Button onClick={startMobileCamera} className="start-camera-btn">
                            <Camera size={18} />
                            Start Detection
                          </Button>
                        ) : (
                          <Button onClick={stopCamera} variant="destructive">
                            Stop Camera
                          </Button>
                        )}
                      </div>
                    </>
                  )}
                  
                  <div className="video-container">
                    {cameraSource === "local" ? (
                      <>
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          className="camera-feed"
                          data-testid="camera-video"
                        />
                        {cameraActive && (
                          <div className="camera-overlay">
                            <div className="fps-counter">Frame: {frameCountRef.current}</div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="mobile-stream-display">
                        {result && result.image ? (
                          <>
                            <img src={result.image} alt="Mobile Stream" className="mobile-stream-img" />
                            {cameraActive && (
                              <div className="camera-overlay">
                                <div className="fps-counter mobile-fps">📱 Mobile | Frame: {frameCountRef.current}</div>
                              </div>
                            )}
                          </>
                        ) : (
                          <div className="stream-placeholder">
                            <Smartphone size={64} />
                            <p>{mobileConnected ? "Waiting for frames..." : "Connect mobile camera to start"}</p>
                          </div>
                        )}
                      </div>
                    )}
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

          {result && result.image && mode === "image" && (
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
                    <span>Distance</span>
                    <span>Status</span>
                  </div>
                  {result.detections.map((detection, idx) => {
                    const statusColor = getStatusColor(detection);
                    return (
                      <div key={idx} className={`table-row status-${statusColor}`} data-testid={`detection-row-${idx}`}>
                        <span className="object-name">{detection.class}</span>
                        <span className="distance-value">
                          {detection.distance ? `${detection.distance}m` : 'N/A'}
                        </span>
                        <span className={`status-badge status-${statusColor}`}>
                          {statusColor === 'critical' && (
                            <><AlertTriangle size={14} /> CRITICAL</>
                          )}
                          {statusColor === 'approaching' && (
                            <><TrendingDown size={14} /> APPROACHING</>
                          )}
                          {statusColor === 'warning' && (
                            <><Bell size={14} /> NEARBY</>
                          )}
                          {statusColor === 'safe' && 'SAFE'}
                        </span>
                      </div>
                    );
                  })}
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

          <Card className="legend-card">
            <CardHeader>
              <CardTitle>Alert Thresholds</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="legend-list">
                <div className="legend-item critical">
                  <AlertTriangle size={16} />
                  <span>Critical: &lt;2m - Voice alert triggered</span>
                </div>
                <div className="legend-item approaching">
                  <TrendingDown size={16} />
                  <span>Approaching: Object moving closer</span>
                </div>
                <div className="legend-item warning">
                  <Bell size={16} />
                  <span>Nearby: Within 5m range</span>
                </div>
              </div>
            </CardContent>
          </Card>

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