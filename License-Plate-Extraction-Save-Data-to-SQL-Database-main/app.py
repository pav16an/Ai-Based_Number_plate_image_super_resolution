from flask import Flask, render_template, request, jsonify, send_file
import json
import cv2
import base64
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
import time
import easyocr
import uuid
import torch
from werkzeug.utils import secure_filename

# Make Real-ESRGAN optional
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    print("Real-ESRGAN not available. Using basic image enhancement instead.")
    REALESRGAN_AVAILABLE = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['DATABASE'] = os.path.join(os.path.dirname(__file__), 'plates.db')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure required directories exist with correct permissions
for directory in ['uploads', 'json', 'data']:
    dir_path = os.path.join(os.path.dirname(__file__), directory)
    os.makedirs(dir_path, exist_ok=True)
    # Ensure directory is writable
    os.chmod(dir_path, 0o777)

def init_db():
    try:
        print(f"Initializing database at: {app.config['DATABASE']}")
        
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(app.config['DATABASE'])
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        with sqlite3.connect(app.config['DATABASE']) as conn:
            # Create plates table
            conn.execute('''CREATE TABLE IF NOT EXISTS plates
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      plate_number TEXT NOT NULL,
                      plate_type TEXT NOT NULL,
                      confidence REAL NOT NULL,
                      source TEXT NOT NULL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            
            # Create plate_images table
            conn.execute('''CREATE TABLE IF NOT EXISTS plate_images
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      plate_id INTEGER NOT NULL,
                      filename TEXT NOT NULL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (plate_id) REFERENCES plates(id))''')
            
            # Create actions table to track all operations
            conn.execute('''CREATE TABLE IF NOT EXISTS actions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      action_type TEXT NOT NULL,
                      plate_id INTEGER,
                      details TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (plate_id) REFERENCES plates(id))''')
            
            conn.commit()
            print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

def save_to_db(plate_number, plate_type, confidence, source, plate_image=None):
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            # Insert plate data
            cursor = conn.execute("""
                INSERT INTO plates (plate_number, plate_type, confidence, source) 
                VALUES (?, ?, ?, ?)
            """, (plate_number, plate_type, confidence, source))
            
            plate_id = cursor.lastrowid
            
            # Log the detection action
            conn.execute("""
                INSERT INTO actions (action_type, plate_id, details)
                VALUES (?, ?, ?)
            """, ('detection', plate_id, f'Detected plate {plate_number} with confidence {confidence:.2f}'))
            
            # Save plate image if provided
            if plate_image is not None:
                try:
                    # Generate unique filename
                    filename = f"plate_{plate_id}_{int(time.time())}.jpg"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    # Ensure the uploads directory exists
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # Save the image
                    success = cv2.imwrite(filepath, plate_image)
                    if not success:
                        raise Exception("Failed to save image")
                    
                    # Save image record to database
                    conn.execute("""
                        INSERT INTO plate_images (plate_id, filename)
                        VALUES (?, ?)
                    """, (plate_id, filename))
                    
                    # Log the image save action
                    conn.execute("""
                        INSERT INTO actions (action_type, plate_id, details)
                        VALUES (?, ?, ?)
                    """, ('image_save', plate_id, f'Saved image {filename}'))
                    
                    print(f"Saved plate image to: {filepath}")
                except Exception as e:
                    print(f"Error saving plate image: {str(e)}")
                    # Log the error action
                    conn.execute("""
                        INSERT INTO actions (action_type, plate_id, details)
                        VALUES (?, ?, ?)
                    """, ('error', plate_id, f'Error saving image: {str(e)}'))
                    # Continue without the image if there's an error
            
            conn.commit()
            print(f"Saved to database: {plate_number} ({plate_type})")
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        import traceback
        traceback.print_exc()

# Initialize models
try:
    weights_path = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file not found at {weights_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO(weights_path)
    yolo_model.to(device)
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    className = ["License"]
    
    # Initialize Real-ESRGAN if available
    if REALESRGAN_AVAILABLE:
        try:
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            
            # Use half precision only on GPU
            use_half = device == 'cuda'
            
            upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=esrgan_model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=use_half,  # Use half precision only on GPU
                device=device
            )
            print(f"Real-ESRGAN initialized successfully on {device} with half precision: {use_half}")
        except Exception as e:
            print(f"Error initializing Real-ESRGAN: {str(e)}")
            REALESRGAN_AVAILABLE = False
    
    # Warm up models
    dummy_input = torch.zeros(1, 3, 640, 640).to(device)
    yolo_model.predict(dummy_input)
    print("Models initialized successfully on", device)
except Exception as e:
    print(f"Error initializing models: {str(e)}")
    raise

def enhance_image(image):
    """Enhance image quality specifically for license plates with advanced preprocessing and enhancement"""
    try:
        # Input validation
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Check image dimensions
        height, width = image.shape[:2]
        if width == 0 or height == 0:
            raise ValueError("Invalid image dimensions")
        
        # Initial preprocessing
        preprocessed = preprocess_image(image)
        
        # Use Real-ESRGAN if available
        if REALESRGAN_AVAILABLE:
            try:
                print("Applying Real-ESRGAN super-resolution...")
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                
                # Enhance image with 4x upscaling for better detail
                output, _ = upsampler.enhance(img_rgb, outscale=4)
                print(f"Super-resolution applied: {preprocessed.shape} -> {output.shape}")
                
                # Convert back to BGR
                enhanced = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                
                # Apply additional enhancements to the upscaled image
                enhanced = apply_plate_enhancements(enhanced)
                
                return enhanced
            except Exception as e:
                print(f"Real-ESRGAN enhancement failed: {str(e)}")
                # Fall back to basic enhancement
        
        print("Using basic enhancement as fallback...")
        # Basic enhancement as fallback
        return apply_plate_enhancements(preprocessed)
        
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        # Log the error for debugging
        import traceback
        traceback.print_exc()
        return image  # Return original image on failure

def preprocess_image(image):
    """Apply initial preprocessing to improve image quality"""
    try:
        # Convert to grayscale for initial processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2,2), np.uint8)
        enhanced_bgr = cv2.morphologyEx(enhanced_bgr, cv2.MORPH_CLOSE, kernel)
        enhanced_bgr = cv2.morphologyEx(enhanced_bgr, cv2.MORPH_OPEN, kernel)
        
        return enhanced_bgr
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return image

def apply_plate_enhancements(image):
    """Apply advanced enhancements specifically for license plates"""
    try:
        print("Applying plate-specific enhancements...")
        
        # Convert to LAB color space for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels back
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply advanced sharpening
        kernel_sharpen = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Apply contrast enhancement with adaptive parameters
        mean_val = np.mean(enhanced)
        if mean_val < 100:  # Dark image
            alpha = 1.8  # Higher contrast
            beta = 20    # Higher brightness
        elif mean_val > 200:  # Bright image
            alpha = 1.2  # Lower contrast
            beta = -10   # Lower brightness
        else:  # Normal image
            alpha = 1.5  # Moderate contrast
            beta = 10    # Moderate brightness
            
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # Apply bilateral filter with adaptive parameters
        d = 9  # Diameter of each pixel neighborhood
        sigma_color = 75  # Filter sigma in the color space
        sigma_space = 75  # Filter sigma in the coordinate space
        enhanced = cv2.bilateralFilter(enhanced, d, sigma_color, sigma_space)
        
        # Edge enhancement for text
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        enhanced = cv2.addWeighted(enhanced, 1.0, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)
        
        # Final noise reduction
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        print("Plate enhancements applied successfully")
        return enhanced
    except Exception as e:
        print(f"Error in plate enhancements: {str(e)}")
        return image

def detect_plate_type(frame):
    """Detect license plate type based on color using advanced color analysis"""
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Reshape the image to be a list of pixels
        pixels = hsv.reshape(-1, 3)
        
        # Convert to float32
        pixels = np.float32(pixels)
        
        # Define criteria for k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Perform k-means clustering
        K = 3  # Number of clusters
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Get the dominant colors and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_colors = centers[unique_labels]
        
        # Sort colors by frequency
        color_freq = list(zip(dominant_colors, counts))
        color_freq.sort(key=lambda x: x[1], reverse=True)
        
        # Get the most dominant color
        dominant_hsv = color_freq[0][0]
        hue, saturation, value = dominant_hsv
        
        # Calculate color confidence based on cluster distribution
        total_pixels = sum(counts)
        color_confidence = color_freq[0][1] / total_pixels
        
        # Define color ranges with confidence adjustments
        color_ranges = [
            # Black plates (Commercial)
            {
                'range': lambda h, s, v: v < 50 and s < 50,
                'type': "Commercial (Black)",
                'confidence_multiplier': 1.2 if v < 30 else 0.8
            },
            # White plates (Private)
            {
                'range': lambda h, s, v: s < 50 and v > 200,
                'type': "Private (White)",
                'confidence_multiplier': 1.2 if v > 220 else 0.8
            },
            # Yellow plates (Commercial)
            {
                'range': lambda h, s, v: 20 <= h <= 30 and s > 100,
                'type': "Commercial (Yellow)",
                'confidence_multiplier': 1.2 if 22 <= h <= 28 else 0.8
            },
            # Green plates (Electric)
            {
                'range': lambda h, s, v: 50 <= h <= 70 and s > 100,
                'type': "Electric (Green)",
                'confidence_multiplier': 1.2 if 55 <= h <= 65 else 0.8
            },
            # Blue plates (Diplomatic)
            {
                'range': lambda h, s, v: 100 <= h <= 140 and s > 100,
                'type': "Diplomatic (Blue)",
                'confidence_multiplier': 1.2 if 110 <= h <= 130 else 0.8
            },
            # Red plates (Temporary)
            {
                'range': lambda h, s, v: (h < 10 or h > 170) and s > 100,
                'type': "Temporary (Red)",
                'confidence_multiplier': 1.2 if (h < 5 or h > 175) else 0.8
            },
            # Silver plates (Private)
            {
                'range': lambda h, s, v: s < 30 and 150 < v < 200,
                'type': "Private (Silver)",
                'confidence_multiplier': 1.2 if 160 < v < 190 else 0.8
            }
        ]
        
        # Check each color range
        for color_range in color_ranges:
            if color_range['range'](hue, saturation, value):
                # Calculate final confidence
                final_confidence = color_confidence * color_range['confidence_multiplier']
                # Log the detection with confidence
                print(f"Detected {color_range['type']} with confidence {final_confidence:.2f}")
                return color_range['type'], final_confidence
        
        # If no specific color is detected, check for grayscale plates
        if saturation < 30:
            if v > 200:
                return "Private (White)", color_confidence * 0.9
            elif v < 50:
                return "Commercial (Black)", color_confidence * 0.9
            else:
                return "Private (Silver)", color_confidence * 0.9
        
        return "Unknown", color_confidence
        
    except Exception as e:
        print(f"Error detecting plate type: {str(e)}")
        return "Unknown", 0.0

def process_frame(frame):
    """Process a single frame and detect license plates"""
    try:
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame received")
            
        # Ensure frame has valid dimensions
        height, width = frame.shape[:2]
        if width == 0 or height == 0:
            raise ValueError("Invalid frame dimensions")
            
        # Resize frame to a standard size while maintaining aspect ratio
        max_dim = 1280
        scale = min(max_dim / width, max_dim / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection with adjusted parameters
        results = yolo_model.predict(
            frame_rgb,
            conf=0.25,  # Lower confidence threshold
            iou=0.3,    # Lower IoU threshold
            max_det=10  # Maximum number of detections
        )
        
        detected_plates = []
        enhanced_frame = frame.copy()  # Create a copy for drawing
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Extract plate region with padding
                padding = 5
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    continue
                
                print(f"Processing license plate region: {plate_img.shape}")
                
                # Apply super-resolution to the license plate region
                enhanced_plate = enhance_image(plate_img)
                print(f"Enhanced plate region: {enhanced_plate.shape}")
                
                # Save for debugging
                plate_path = os.path.join(app.config['UPLOAD_FOLDER'], f'plate_{x1}_{y1}.jpg')
                cv2.imwrite(plate_path, enhanced_plate)
                print(f"Saved enhanced plate to: {plate_path}")
                
                # Preprocess plate image for better OCR
                gray_plate = cv2.cvtColor(enhanced_plate, cv2.COLOR_BGR2GRAY)
                blur_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
                thresh_plate = cv2.adaptiveThreshold(
                    blur_plate,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
                )
                
                # Detect plate type
                plate_type, plate_confidence = detect_plate_type(enhanced_plate)
                
                # OCR with multiple attempts
                try:
                    # First attempt with enhanced plate
                    ocr_result = reader.readtext(enhanced_plate)
                    
                    # Second attempt with preprocessed plate
                    if not ocr_result:
                        ocr_result = reader.readtext(thresh_plate)
                    
                    # Third attempt with inverted colors
                    if not ocr_result:
                        inverted = cv2.bitwise_not(thresh_plate)
                        ocr_result = reader.readtext(inverted)
                    
                    if ocr_result:
                        text = ocr_result[0][1]
                        confidence = ocr_result[0][2]
                        
                        # Clean and correct text
                        text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        text = (text.replace('O', '0')
                                   .replace('I', '1')
                                   .replace('Z', '2')
                                   .replace('S', '5')
                                   .replace('B', '8')
                                   .replace('G', '6')
                                   .replace('T', '7'))
                        
                        if text and len(text) >= 4:
                            # Combine OCR and color detection confidences
                            combined_confidence = (confidence + plate_confidence) / 2
                            
                            detected_plates.append({
                                'number': text,
                                'type': plate_type,
                                'confidence': combined_confidence,
                                'box': [x1, y1, x2, y2]
                            })
                            # Save to database with the enhanced plate image
                            save_to_db(text, plate_type, combined_confidence, 'image', enhanced_plate)
                            
                            # Draw detection on frame with color-coded confidence
                            color = (0, int(255 * combined_confidence), 0)  # Green with intensity based on confidence
                            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(enhanced_frame, f"{text} ({combined_confidence:.2f})",
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, color, 2)
                            
                            # Replace the original plate region with the enhanced one
                            enhanced_frame[y1:y2, x1:x2] = enhanced_plate
                            print(f"Detected plate: {text} ({combined_confidence:.2f}) - {plate_type}")
                except Exception as e:
                    print(f"OCR error: {str(e)}")
                    continue
                
        return detected_plates, enhanced_frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return [], frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/database')
def view_database():
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.cursor()
            # Get all plates with their images
            cursor.execute("""
                SELECT p.id, p.plate_number, p.plate_type, p.confidence, p.source, p.timestamp,
                       (SELECT filename FROM plate_images WHERE plate_id = p.id ORDER BY timestamp DESC LIMIT 1) as image
                FROM plates p
                ORDER BY p.timestamp DESC
            """)
            plates = cursor.fetchall()
            
            # Get statistics
            cursor.execute("SELECT COUNT(*) FROM plates")
            total_plates = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM plates")
            unique_plates = cursor.fetchone()[0]
            
            cursor.execute("SELECT plate_type, COUNT(*) FROM plates GROUP BY plate_type")
            plate_types = cursor.fetchall()
            
            # Get recent actions
            cursor.execute("""
                SELECT a.action_type, a.details, a.timestamp, p.plate_number
                FROM actions a
                LEFT JOIN plates p ON a.plate_id = p.id
                ORDER BY a.timestamp DESC
                LIMIT 20
            """)
            recent_actions = cursor.fetchall()
            
            return render_template('database.html', 
                                plates=plates,
                                total_plates=total_plates,
                                unique_plates=unique_plates,
                                plate_types=plate_types,
                                recent_actions=recent_actions)
    except Exception as e:
        print(f"Database error: {str(e)}")
        return render_template('database.html', 
                             plates=[],
                             total_plates=0,
                             unique_plates=0,
                             plate_types=[],
                             recent_actions=[],
                             error=str(e))

@app.route('/plate_image/<int:plate_id>')
def get_plate_image(plate_id):
    """Serve the most recent image for a plate"""
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT filename FROM plate_images 
                WHERE plate_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (plate_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], result[0])
                if os.path.exists(image_path):
                    return send_file(image_path)
            
            # Return 204 No Content if no image is available
            return '', 204
    except Exception as e:
        print(f"Error serving plate image: {str(e)}")
        # Return 204 No Content on error
        return '', 204

@app.route('/api/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        # Save uploaded file
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process based on file type
        if file.content_type.startswith('image/'):
            # Process image
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError("Failed to read image file")
                
            detected_plates, enhanced_image = process_frame(image)
            
            # Save processed image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            cv2.imwrite(output_path, enhanced_image)
            
            return jsonify({
                'license_plates': detected_plates,
                'image': f'/uploads/processed_{filename}'
            })
            
        elif file.content_type.startswith('video/'):
            # Start video processing in background
            # For now, just process first frame
            cap = cv2.VideoCapture(filepath)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Failed to read video file")
                
            detected_plates, enhanced_frame = process_frame(frame)
            
            # Save processed frame
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}.jpg')
            cv2.imwrite(output_path, enhanced_frame)
            
            return jsonify({
                'license_plates': detected_plates,
                'image': f'/uploads/processed_{filename}.jpg',
                'message': 'Video processing started'
            })
            
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/webcam', methods=['POST'])
def process_webcam():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
            
        # Ensure frame has valid dimensions
        height, width = frame.shape[:2]
        if width == 0 or height == 0:
            return jsonify({'error': 'Invalid frame dimensions'}), 400
            
        # Process frame with enhancement
        detected_plates, enhanced_frame = process_frame(frame)
        
        # Convert frame back to base64 for display
        _, buffer = cv2.imencode('.jpg', enhanced_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'license_plates': detected_plates,
            'processed_image': f'data:image/jpeg;base64,{processed_image}'
        })
        
    except Exception as e:
        print(f"Error in process_webcam: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates', methods=['GET'])
def get_plates():
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plates ORDER BY timestamp DESC")
            plates = cursor.fetchall()
            return jsonify([{
                'id': p[0],
                'number': p[1],
                'type': p[2],
                'confidence': p[3],
                'source': p[4],
                'timestamp': p[5]
            } for p in plates])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates/<int:plate_id>', methods=['DELETE'])
def delete_plate(plate_id):
    try:
        with sqlite3.connect(app.config['DATABASE']) as conn:
            # First get plate information for logging
            cursor = conn.cursor()
            cursor.execute("SELECT plate_number, plate_type FROM plates WHERE id = ?", (plate_id,))
            plate_info = cursor.fetchone()
            
            if not plate_info:
                return jsonify({'error': 'Plate not found'}), 404
                
            plate_number, plate_type = plate_info
            
            # Get all associated image filenames
            cursor.execute("SELECT filename FROM plate_images WHERE plate_id = ?", (plate_id,))
            image_files = cursor.fetchall()
            
            # Log the deletion action before deleting
            conn.execute("""
                INSERT INTO actions (action_type, plate_id, details)
                VALUES (?, ?, ?)
            """, ('deletion', plate_id, f'Deleted plate {plate_number} ({plate_type})'))
            
            # Delete the images from the uploads directory
            for image_file in image_files:
                try:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file[0])
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted image file: {file_path}")
                        
                        # Log each image deletion
                        conn.execute("""
                            INSERT INTO actions (action_type, details)
                            VALUES (?, ?)
                        """, ('file_deletion', f'Deleted image file: {image_file[0]}'))
                except Exception as e:
                    print(f"Error deleting image file {image_file[0]}: {str(e)}")
                    # Log the error
                    conn.execute("""
                        INSERT INTO actions (action_type, details)
                        VALUES (?, ?)
                    """, ('error', f'Error deleting image file {image_file[0]}: {str(e)}'))
            
            # Delete the plate record (this will cascade delete the image records)
            conn.execute("DELETE FROM plates WHERE id = ?", (plate_id,))
            conn.commit()
            
            return jsonify({'message': 'Plate deleted successfully'})
    except Exception as e:
        print(f"Error deleting plate: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def process_video(video_path):
    """Process video file and detect license plates with improved handling"""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}, Total frames: {total_frames}")
        
        # Create output video writer with improved codec
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{os.path.basename(video_path)}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Initialize tracking variables
        frame_count = 0
        processed_frames = 0
        detected_plates = []
        last_detection_time = 0
        detection_interval = 1.0  # Minimum time between detections in seconds
        error_frames = 0
        max_error_frames = 10  # Maximum number of consecutive error frames before skipping
        
        # Create progress tracking file
        progress_file = os.path.join(app.config['UPLOAD_FOLDER'], 'video_progress.json')
        progress_data = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'detected_plates': [],
            'status': 'processing',
            'start_time': time.time()
        }
        
        def update_progress():
            progress_data['processed_frames'] = processed_frames
            progress_data['detected_plates'] = detected_plates
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                    
                frame_count += 1
                current_time = frame_count / fps
                
                # Skip frames if too many errors
                if error_frames >= max_error_frames:
                    print(f"Skipping frame {frame_count} due to too many errors")
                    error_frames = 0
                    continue
                
                # Process frame with error handling
                try:
                    plates, enhanced_frame = process_frame(frame)
                    
                    # Update detections if enough time has passed
                    if current_time - last_detection_time >= detection_interval:
                        if plates:
                            # Filter out duplicate plates in the interval
                            new_plates = [p for p in plates if p not in detected_plates]
                            detected_plates.extend(new_plates)
                            last_detection_time = current_time
                            print(f"Frame {frame_count}: Detected {len(new_plates)} new plates")
                    
                    # Write the enhanced frame
                    out.write(enhanced_frame)
                    processed_frames += 1
                    error_frames = 0  # Reset error counter on success
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    error_frames += 1
                    # Write original frame if processing fails
                    out.write(frame)
                    processed_frames += 1
                
                # Update progress periodically
                if frame_count % 30 == 0:  # Update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                    update_progress()
                
            except Exception as e:
                print(f"Error reading frame {frame_count}: {str(e)}")
                error_frames += 1
                continue
        
        # Release resources
        cap.release()
        out.release()
        
        # Update final progress
        progress_data['status'] = 'completed'
        progress_data['end_time'] = time.time()
        progress_data['processing_time'] = progress_data['end_time'] - progress_data['start_time']
        update_progress()
        
        print(f"Video processing completed. Processed {processed_frames} frames.")
        print(f"Total unique plates detected: {len(set(p['number'] for p in detected_plates))}")
        print(f"Output saved to: {output_path}")
        
        return detected_plates, output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update progress file with error status
        if 'progress_data' in locals():
            progress_data['status'] = 'error'
            progress_data['error'] = str(e)
            update_progress()
        
        return [], None

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload and process it with improved error handling"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
            
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Validate video format
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Invalid video format. Supported formats: {", ".join(allowed_extensions)}'
            }), 400
            
        # Generate unique filename
        filename = secure_filename(f"{uuid.uuid4()}{file_ext}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the video file
        video_file.save(video_path)
        
        # Validate video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({'error': 'Invalid or corrupted video file'}), 400
        cap.release()
        
        # Process the video
        detected_plates, output_path = process_video(video_path)
        
        if not detected_plates:
            return jsonify({'error': 'No license plates detected in the video'}), 400
            
        # Get unique plates
        unique_plates = []
        seen_numbers = set()
        for plate in detected_plates:
            if plate['number'] not in seen_numbers:
                seen_numbers.add(plate['number'])
                unique_plates.append(plate)
        
        # Clean up original video file
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Warning: Could not remove original video file: {str(e)}")
        
        # Return results
        return jsonify({
            'message': 'Video processed successfully',
            'detected_plates': unique_plates,
            'total_plates': len(unique_plates),
            'output_video': os.path.basename(output_path) if output_path else None,
            'processing_time': progress_data.get('processing_time', 0)
        })
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/video_progress')
def get_video_progress():
    """Get the current progress of video processing"""
    try:
        progress_file = os.path.join(app.config['UPLOAD_FOLDER'], 'video_progress.json')
        if not os.path.exists(progress_file):
            return jsonify({'error': 'No video processing in progress'}), 404
            
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            
        return jsonify(progress_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest_image')
def get_latest_image():
    try:
        # Get the list of all files in the uploads directory
        uploads_dir = os.path.join(app.root_path, 'uploads')
        if not os.path.exists(uploads_dir):
            return jsonify({'error': 'Uploads directory not found'}), 404
            
        # Get all image files
        image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            return jsonify({'error': 'No images found'}), 404
            
        # Sort files by modification time (newest first)
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)), reverse=True)
        
        # Get the most recent image
        latest_image = image_files[0]
        image_path = f'/uploads/{latest_image}'
        
        return jsonify({
            'image_path': image_path,
            'timestamp': os.path.getmtime(os.path.join(uploads_dir, latest_image))
        })
    except Exception as e:
        print(f"Error getting latest image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
