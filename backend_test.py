import requests
import sys
import os
from datetime import datetime
from io import BytesIO
from PIL import Image
import tempfile

class YOLODriveAssistTester:
    def __init__(self, base_url="https://yolo-drive.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, response_type='json'):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, headers=headers, timeout=60)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=60)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                
                if response_type == 'json':
                    try:
                        response_data = response.json()
                        print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                        return True, response_data
                    except:
                        print(f"   Response: {response.text[:200]}...")
                        return True, {}
                else:
                    print(f"   Response size: {len(response.content)} bytes")
                    return True, response.content
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Error: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_image(self):
        """Create a simple test image"""
        img = Image.new('RGB', (640, 480), color='blue')
        img_buffer = BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer

    def create_test_video(self):
        """Create a simple test video file (mock)"""
        # For testing purposes, we'll create a small file
        video_buffer = BytesIO()
        video_buffer.write(b'fake_video_data_for_testing')
        video_buffer.seek(0)
        return video_buffer

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        return success

    def test_image_detection(self):
        """Test image detection endpoint"""
        test_image = self.create_test_image()
        
        success, response = self.run_test(
            "Image Detection",
            "POST",
            "detect/image",
            200,
            data={
                'gps_lat': 12.9716,
                'gps_lon': 77.5946
            },
            files={'file': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        if success and response:
            # Check response structure
            expected_keys = ['image', 'detections', 'alert_triggered']
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                print(f"   ⚠️  Missing response keys: {missing_keys}")
            else:
                print(f"   ✅ Response structure correct")
                print(f"   Detections count: {len(response.get('detections', []))}")
                print(f"   Alert triggered: {response.get('alert_triggered', False)}")
        
        return success

    def test_frame_detection(self):
        """Test frame detection endpoint (for live camera)"""
        test_image = self.create_test_image()
        
        success, response = self.run_test(
            "Frame Detection",
            "POST",
            "detect/frame",
            200,
            data={
                'gps_lat': 12.9716,
                'gps_lon': 77.5946
            },
            files={'file': ('frame.jpg', test_image, 'image/jpeg')}
        )
        
        if success and response:
            # Check response structure
            expected_keys = ['image', 'detections', 'alert_triggered', 'alerts']
            missing_keys = [key for key in expected_keys if key not in response]
            if missing_keys:
                print(f"   ⚠️  Missing response keys: {missing_keys}")
            else:
                print(f"   ✅ Response structure correct")
                print(f"   Detections count: {len(response.get('detections', []))}")
                print(f"   Alerts count: {len(response.get('alerts', []))}")
        
        return success

    def test_video_detection(self):
        """Test video detection endpoint"""
        test_video = self.create_test_video()
        
        success, response = self.run_test(
            "Video Detection",
            "POST",
            "detect/video",
            200,
            data={
                'gps_lat': 12.9716,
                'gps_lon': 77.5946
            },
            files={'file': ('test.mp4', test_video, 'video/mp4')},
            response_type='binary'
        )
        
        return success

    def test_get_alerts(self):
        """Test get alerts endpoint"""
        success, response = self.run_test(
            "Get Alerts",
            "GET",
            "alerts",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   ✅ Alerts retrieved: {len(response)} alerts")
            if response:
                alert = response[0]
                expected_keys = ['id', 'timestamp', 'object_class', 'distance', 'message']
                missing_keys = [key for key in expected_keys if key not in alert]
                if missing_keys:
                    print(f"   ⚠️  Missing alert keys: {missing_keys}")
                else:
                    print(f"   ✅ Alert structure correct")
        
        return success

    def test_get_logs(self):
        """Test get detection logs endpoint"""
        success, response = self.run_test(
            "Get Detection Logs",
            "GET",
            "logs",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   ✅ Logs retrieved: {len(response)} logs")
            if response:
                log = response[0]
                expected_keys = ['id', 'timestamp', 'detections', 'alert_triggered']
                missing_keys = [key for key in expected_keys if key not in log]
                if missing_keys:
                    print(f"   ⚠️  Missing log keys: {missing_keys}")
                else:
                    print(f"   ✅ Log structure correct")
        
        return success

def main():
    print("🚀 Starting YOLOv8 Driver Assistance API Tests")
    print("=" * 60)
    
    # Setup
    tester = YOLODriveAssistTester()
    
    # Run all tests
    tests = [
        tester.test_root_endpoint,
        tester.test_image_detection,
        tester.test_frame_detection,
        tester.test_get_alerts,
        tester.test_get_logs,
        # Note: Video test might fail with fake data, so we'll run it last
        tester.test_video_detection,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    # Print results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())