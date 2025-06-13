import cv2
import time
from typing import Dict, List, Tuple

def get_class_colors() -> Dict[str, str]:
    return {
        'Pesawat': '#FF6B6B',  
        'Burung': '#4ECDC4',   
        'Drone': '#45B7D1',    
        'Helikopter': '#96CEB4'  
    }

def get_class_icons() -> Dict[str, str]:
    return {
        'Pesawat': '✈️',
        'Burung': '🐦',
        'Drone': '🛸',
        'Helikopter': '🚁'
    }

class FPSCalculator:
    """Helper class to calculate FPS"""
    
    def __init__(self, buffer_size: int = 15):
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(frame_time)
        
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        
        return 0.0
    
    def reset(self):
        """Reset FPS calculator"""
        self.frame_times = []
        self.last_time = time.time()

class DetectionCounter:    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.session_counts = {'Pesawat': 0, 'Burung': 0, 'Drone': 0, 'Helikopter': 0}
        self.total_counts = {'Pesawat': 0, 'Burung': 0, 'Drone': 0, 'Helikopter': 0}
        self.frame_counts = {'Pesawat': 0, 'Burung': 0, 'Drone': 0, 'Helikopter': 0}
    
    def update_frame_counts(self, detections: List[Dict]):
        """Update counts for current frame"""
        for key in self.frame_counts:
            self.frame_counts[key] = 0
        for detection in detections:
            class_name = detection.get('class_name', '')
            if class_name in self.frame_counts:
                self.frame_counts[class_name] += 1
    
    def update_session_counts(self, detections: List[Dict]):
        """Update session total counts"""
        for detection in detections:
            class_name = detection.get('class_name', '')
            if class_name in self.session_counts:
                self.session_counts[class_name] += 1
    
    def get_frame_summary(self) -> str:
        total = sum(self.frame_counts.values())
        if total == 0:
            return "Tidak ada deteksi"
        
        summary_parts = []
        icons = get_class_icons()
        
        for class_name, count in self.frame_counts.items():
            if count > 0:
                icon = icons.get(class_name, '❓')
                summary_parts.append(f"{icon} {class_name}: {count}")
        
        return f"Terdeteksi {total} objek: " + ", ".join(summary_parts)
    
    def get_session_summary(self) -> str:
        """Get summary of session total detections"""
        total = sum(self.session_counts.values())
        if total == 0:
            return "Belum ada deteksi dalam sesi ini"
        
        summary_parts = []
        icons = get_class_icons()
        
        for class_name, count in self.session_counts.items():
            if count > 0:
                icon = icons.get(class_name, '❓')
                summary_parts.append(f"{icon} {class_name}: {count}")
        
        return f"Total sesi: {total} deteksi - " + ", ".join(summary_parts)

def format_confidence(confidence: float) -> str:
    """Format confidence score for display"""
    return f"{confidence:.1%}"

def format_bbox(bbox: List[float]) -> str:
    """Format bounding box coordinates for display"""
    if len(bbox) >= 4:
        return f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
    return "Invalid bbox"

def check_camera_available(camera_index: int) -> Tuple[bool, str]:
    """Check if camera is available"""
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return True, f"Kamera {camera_index} tersedia"
            else:
                return False, f"Kamera {camera_index} tidak dapat membaca frame"
        else:
            return False, f"Tidak dapat membuka kamera {camera_index}"
    except Exception as e:
        return False, f"Error checking camera {camera_index}: {str(e)}"

def get_optimal_camera_settings(camera_index: int) -> Dict[str, int]:
    """Get optimal camera settings"""
    settings = {
        'width': 640,
        'height': 480,
        'fps': 15
    }
    
    try:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if current_width >= 320 and current_height >= 240:
                settings['width'] = min(current_width, 1280)  
                settings['height'] = min(current_height, 720)   
            
            if current_fps > 0 and current_fps <= 60:
                settings['fps'] = current_fps
            
            cap.release()
    except Exception:
        pass 
    
    return settings

def apply_camera_settings(cap: cv2.VideoCapture, settings: Dict[str, int]) -> bool:
    """Apply camera settings"""
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
        cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        
  
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True
    except Exception as e:
        print(f"Warning: Could not apply camera settings: {e}")
        return False