import os
import cv2
import shutil
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
input_folder = "/Users/joe/Code/photo-organizer/source"  # ← change this
output_folder_base = "/Users/joe/Code/photo-organizer"  # Base folder for outputs
black_threshold = 30        # Pixel value below which it's considered "black"
black_ratio_cutoff = 0.9    # 90% black = flag the image
white_threshold = 225       # Pixel value above this is "white"
white_ratio_cutoff = 0.9    # Flag if 90% of pixels are white
resize_dims = (192, 108)    # Downscale for speed; preserves aspect ratio
blur_threshold = 300.0  # Lower = more blurry; tweak as needed

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

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Photo organizer - detect and move unwanted images')
    parser.add_argument('check_type', choices=['black', 'white', 'blurry'], 
                       help='Type of image check to perform')
    parser.add_argument('--input', '-i', default=input_folder,
                       help='Input folder containing images')
    parser.add_argument('--output', '-o', 
                       help='Output folder (if not specified, uses check_type as folder name)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
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
        'blurry': check_image_blurry
    }
    check_function = check_functions[args.check_type]
    
    # Get list of image files
    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ]

    print(f"Analyzing {len(image_files)} images for {args.check_type} issues...")

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