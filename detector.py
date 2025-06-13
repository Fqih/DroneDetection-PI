import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

class DroneDetector:
    def __init__(self, model_path="Model/YoloV12_Best.pt"):
        """Initialize the drone detector with YOLO model"""
        self.model_path = model_path
        self.model = self._load_model()
        
        # Class mappings
        self.class_names = {
            0: 'Pesawat', 
            1: 'Burung', 
            2: 'Drone', 
            3: 'Helikopter'
        }
        
        # Class colors for visualization
        self.class_colors = {
            'Pesawat': '#FF6B6B',  
            'Burung': '#4ECDC4',    
            'Drone': '#45B7D1',      
            'Helikopter': '#96CEB4'  
        }
        
        # Class icons
        self.class_icons = {
            'Pesawat': '✈️',
            'Burung': '🐦',
            'Drone': '🛸',
            'Helikopter': '🚁'
        }
    
    @st.cache_resource
    def _load_model(_self):
        """Load YOLO model with caching"""
        try:
            model = YOLO(_self.model_path)
            return model
        except Exception as e:
            st.error(f"❌ Gagal memuat model YOLO: {e}")
            st.error(f"Pastikan file model ada di: {_self.model_path}")
            return None
    
    def detect(self, frame, confidence_threshold=0.5):
        """Run detection on a single frame"""
        if self.model is None:
            return None
        
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            return results
        except Exception as e:
            st.error(f"Error dalam deteksi: {e}")
            return None
    
    def process_results(self, results):
        """Process YOLO results and return annotated frame and detection info"""
        if results is None or len(results) == 0:
            return None, []
        
        try:
            result = results[0]
            
            annotated_frame = result.plot()
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract detection data
                    conf = float(box.conf.item())
                    cls_id = int(box.cls.item())
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = self.class_names.get(cls_id, f"Class_{cls_id}")
                    
                    detection_info = {
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox.tolist(),
                        'class_id': cls_id
                    }
                    
                    detections.append(detection_info)
            
            return annotated_frame, detections
            
        except Exception as e:
            st.error(f"Error memproses hasil: {e}")
            return None, []
    
    def get_detection_summary(self, detections):
        """Get summary of detections by class"""
        if not detections:
            return {}
        
        summary = {}
        for detection in detections:
            class_name = detection['class_name']
            if class_name in summary:
                summary[class_name] += 1
            else:
                summary[class_name] = 1
        
        return summary
    
    def draw_custom_annotations(self, frame, detections):
        """Draw custom annotations on frame (alternative to YOLO's built-in plot)"""
        if not detections:
            return frame
        
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            color_hex = self.class_colors.get(class_name, '#FFFFFF')
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            label = f"{class_name} {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color_bgr, -1)
            
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def is_model_loaded(self):
        """Check if model is successfully loaded"""
        return self.model is not None