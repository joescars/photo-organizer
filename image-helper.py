import os
import cv2
import shutil
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    # Try to import objectron for object detection (includes pets)
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        MEDIAPIPE_OBJECT_DETECTION_AVAILABLE = True
    except ImportError:
        MEDIAPIPE_OBJECT_DETECTION_AVAILABLE = False
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_OBJECT_DETECTION_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")

# Try to import YOLOv5 for pet detection
try:
    import torch
    YOLO_AVAILABLE = True
    yolo_model = None  # Will be loaded when needed
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv5 not available. Install with: pip install torch torchvision ultralytics")

# --- CONFIGURATION ---
input_folder = "/Users/joe/Code/photo-organizer/input"  # ← change this
output_folder_base = "/Users/joe/Code/photo-organizer/output"  # Base folder for outputs
black_threshold = 40        # Pixel value below which it's considered "black"
black_ratio_cutoff = 0.85    # 85% black = flag the image
white_threshold = 190        # Pixel value above this is "white"
white_ratio_cutoff = 0.85    # Flag if 85% of pixels are white
resize_dims = (192, 108)    # Downscale for speed; preserves aspect ratio
blur_threshold = 300.0  # Lower = more blurry; tweak as needed

# MediaPipe face detection settings
mediapipe_model_selection = 0      # 0 = short-range (2m), 1 = full-range (5m)
mediapipe_min_detection_confidence = 0.4  # 0.1-1.0, lower = more sensitive
mediapipe_resize_for_detection = None     # Set to (width, height) to resize, None = original size

# Pet detection settings
pet_confidence_threshold = 0.5    # Confidence threshold for pet detection
pet_model_size = 'yolov5s'        # yolov5s, yolov5m, yolov5l, yolov5x (s=smallest/fastest, x=largest/most accurate)

# Initialize face detectors and YOLO model (will be loaded once when needed)
face_cascade = None
mediapipe_detector = None
yolo_model = None
current_detector_type = 'mediapipe'  # Global variable to store detector type

# COCO class names for YOLOv5 (YOLO uses COCO dataset)
YOLO_PET_CLASSES = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow']  # Animal classes from COCO
YOLO_PET_CLASS_IDS = [14, 15, 16, 17, 18, 19]  # Corresponding class IDs in COCO

def set_detector_type(detector_type):
    """Set the global detector type"""
    global current_detector_type
    current_detector_type = detector_type

def get_face_detector(detector_type='haar'):
    """Initialize face detector on first use"""
    global face_cascade, mediapipe_detector
    
    if detector_type == 'haar':
        if face_cascade is None:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    elif detector_type == 'mediapipe':
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        if mediapipe_detector is None:
            mp_face_detection = mp.solutions.face_detection
            mediapipe_detector = mp_face_detection.FaceDetection(
                model_selection=mediapipe_model_selection, 
                min_detection_confidence=mediapipe_min_detection_confidence
            )
        return mediapipe_detector

def get_yolo_model():
    """Initialize YOLO model on first use"""
    global yolo_model
    if not YOLO_AVAILABLE:
        raise ImportError("YOLOv5 not available. Install with: pip install torch torchvision ultralytics")
    
    if yolo_model is None:
        try:
            import ultralytics
            yolo_model = ultralytics.YOLO(f'{pet_model_size}.pt')
        except ImportError:
            # Fallback to torch hub if ultralytics not available
            yolo_model = torch.hub.load('ultralytics/yolov5', pet_model_size, pretrained=True)
            yolo_model.eval()
    return yolo_model

def detect_pets_yolo(image_path):
    """Detect pets using YOLOv5"""
    try:
        if not YOLO_AVAILABLE:
            return detect_pets_basic(image_path)
        
        model = get_yolo_model()
        
        # Run inference
        results = model(image_path)
        
        # Check if using ultralytics YOLO or torch hub YOLO
        if hasattr(results, 'pandas'):
            # torch hub version
            detections = results.pandas().xyxy[0]
            for _, detection in detections.iterrows():
                class_id = int(detection['class'])
                confidence = float(detection['confidence'])
                
                if class_id in YOLO_PET_CLASS_IDS and confidence >= pet_confidence_threshold:
                    return True
        else:
            # ultralytics version
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in YOLO_PET_CLASS_IDS and confidence >= pet_confidence_threshold:
                            return True
        
        return False
        
    except Exception as e:
        print(f"YOLO detection error for {image_path}: {e}")
        # Fallback to basic detection
        return detect_pets_basic(image_path)

def detect_pets_basic(image_path):
    """Basic pet detection using color and edge detection as fallback"""
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return False
        
        # Simple heuristic: look for fur-like textures and animal-like shapes
        # This is a very basic approach and not very accurate
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Look for fur-like textures using Gabor filters
        # This is a simplified approach - real pet detection needs ML models
        kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # If we detect high texture variance, it might indicate fur
        texture_variance = filtered.var()
        
        # Very basic threshold - this is not reliable for real pet detection
        return texture_variance > 1000
        
    except Exception as e:
        return False

def check_image_no_pets(image_path):
    """Check for images without pets (returns path if NO pets found)"""
    try:
        # Use YOLO for pet detection, fallback to basic if not available
        has_pets = detect_pets_yolo(image_path)
        
        if not has_pets:
            return image_path
            
    except Exception as e:
        return image_path  # Error = assume it's bad
    return None

def check_image_has_pets(image_path):
    """Check for images with pets (returns path if pets ARE found)"""
    try:
        # Use YOLO for pet detection, fallback to basic if not available
        has_pets = detect_pets_yolo(image_path)
        
        if has_pets:
            return image_path
            
    except Exception as e:
        return image_path  # Error = assume it's bad
    return None

def check_image_no_faces(image_path, detector_type='mediapipe'):
    """Check for images without faces (returns path if NO faces found)"""
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return image_path  # unreadable or corrupt
        
        if detector_type == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Optional resize for better detection on large images
            if mediapipe_resize_for_detection:
                rgb_img = cv2.resize(rgb_img, mediapipe_resize_for_detection)
            
            detector = get_face_detector('mediapipe')
            results = detector.process(rgb_img)
            
            # Return path if NO faces found
            if results.detections is None or len(results.detections) == 0:
                return image_path
        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = get_face_detector('haar')
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Return path if NO faces found
            if len(faces) == 0:
                return image_path
            
    except Exception as e:
        return image_path  # Error = assume it's bad
    return None

def check_image_has_faces(image_path, detector_type='mediapipe'):
    """Check for images with faces (returns path if faces ARE found)"""
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return image_path  # unreadable or corrupt
        
        if detector_type == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Optional resize for better detection on large images
            if mediapipe_resize_for_detection:
                rgb_img = cv2.resize(rgb_img, mediapipe_resize_for_detection)
            
            detector = get_face_detector('mediapipe')
            results = detector.process(rgb_img)
            
            # Return path if faces ARE found
            if results.detections is not None and len(results.detections) > 0:
                return image_path
        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = get_face_detector('haar')
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Return path if faces ARE found
            if len(faces) > 0:
                return image_path
            
    except Exception as e:
        return image_path  # Error = assume it's bad
    return None

def check_image_black(image_path):
    """Check for mostly black images"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return image_path  # unreadable or corrupt

        img = cv2.resize(img, resize_dims)  # optional for speed

        black_pixels = (img < black_threshold).sum()
        total_pixels = img.size
        black_ratio = black_pixels / total_pixels

        if black_ratio >= black_ratio_cutoff:
            return image_path
    except:
        return image_path  # Error = assume it's bad
    return None

def check_image_white(image_path):
    """Check for mostly white/overexposed images"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return image_path  # unreadable or corrupt

        img = cv2.resize(img, resize_dims)  # Optional resize for performance

        white_pixels = (img > white_threshold).sum()
        total_pixels = img.size
        white_ratio = white_pixels / total_pixels

        if white_ratio >= white_ratio_cutoff:
            return image_path
    except:
        return image_path  # Error = assume it's bad
    return None

def check_image_blurry(image_path):
    """Check for blurry images"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return image_path  # unreadable or corrupt

        img = cv2.resize(img, resize_dims)

        # Blur detection
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

        if laplacian_var < blur_threshold:
            return image_path
    except:
        return image_path  # error or unreadable
    return None

def check_image_no_faces_wrapper(image_path):
    """Wrapper function for multiprocessing - checks for images without faces"""
    return check_image_no_faces(image_path, current_detector_type)

def check_image_has_faces_wrapper(image_path):
    """Wrapper function for multiprocessing - checks for images with faces"""
    return check_image_has_faces(image_path, current_detector_type)

def check_image_no_pets_wrapper(image_path):
    """Wrapper function for multiprocessing - checks for images without pets"""
    return check_image_no_pets(image_path)

def check_image_has_pets_wrapper(image_path):
    """Wrapper function for multiprocessing - checks for images with pets"""
    return check_image_has_pets(image_path)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Photo organizer - detect and move unwanted images')
    parser.add_argument('check_type', choices=['black', 'white', 'blurry', 'no-faces', 'has-faces', 'no-pets', 'has-pets'], 
                       help='Type of image check to perform')
    parser.add_argument('--input', '-i', default=input_folder,
                       help='Input folder containing images')
    parser.add_argument('--output', '-o', 
                       help='Output folder (if not specified, uses check_type as folder name)')
    parser.add_argument('--detector', choices=['haar', 'mediapipe'], default='mediapipe',
                       help='Face detector to use (default: mediapipe)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set the global detector type for multiprocessing
    set_detector_type(args.detector)
    
    # Set up input and output folders
    input_dir = args.input
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.join(output_folder_base, args.check_type)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Select the appropriate check function
    check_functions = {
        'black': check_image_black,
        'white': check_image_white,
        'blurry': check_image_blurry,
        'no-faces': check_image_no_faces_wrapper,
        'has-faces': check_image_has_faces_wrapper,
        'no-pets': check_image_no_pets_wrapper,
        'has-pets': check_image_has_pets_wrapper
    }
    check_function = check_functions[args.check_type]
    
    # Get list of image files
    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ]

    print(f"Analyzing {len(image_files)} images for {args.check_type} issues...")
    if args.check_type in ['no-faces', 'has-faces']:
        detector_name = 'MediaPipe' if args.detector == 'mediapipe' and MEDIAPIPE_AVAILABLE else 'Haar Cascade'
        print(f"Using {detector_name} face detector...")
    elif args.check_type in ['no-pets', 'has-pets']:
        if YOLO_AVAILABLE:
            print(f"Using YOLOv5 ({pet_model_size}) for pet detection...")
        else:
            print("YOLOv5 not available, using basic texture-based pet detection (experimental)...")
            print("For better accuracy, install YOLOv5: pip install torch torchvision ultralytics")

    flagged = []
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(check_function, image_files), total=len(image_files)):
            if result:
                flagged.append(result)

    print(f"\nFlagged {len(flagged)} images.")

    # Move flagged images
    for fpath in flagged:
        try:
            shutil.move(fpath, os.path.join(output_folder, os.path.basename(fpath)))
        except Exception as e:
            print(f"Error moving {fpath}: {e}")
    
    print(f"Moved {len(flagged)} images to {output_folder}")

if __name__ == "__main__":
    main()