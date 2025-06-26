import os
import cv2
import shutil
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Install with: pip install mediapipe")

# --- CONFIGURATION ---
input_folder = "/Users/joe/Code/photo-organizer/source"  # ← change this
output_folder_base = "/Users/joe/Code/photo-organizer"  # Base folder for outputs
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

# Initialize face detectors (will be loaded once when needed)
face_cascade = None
mediapipe_detector = None
current_detector_type = 'mediapipe'  # Global variable to store detector type

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Photo organizer - detect and move unwanted images')
    parser.add_argument('check_type', choices=['black', 'white', 'blurry', 'no-faces', 'has-faces'], 
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
        'has-faces': check_image_has_faces_wrapper
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