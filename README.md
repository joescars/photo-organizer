# Photo Organizer

A Python script for automatically detecting and organizing unwanted photos from your image collection. Uses computer vision techniques to identify blurry, overexposed, or mostly blank images and moves them to separate folders for review.

## Features

- **Blur Detection**: Identifies blurry or out-of-focus images using Laplacian variance
- **Black Image Detection**: Finds mostly black or blank images (configurable threshold)
- **White/Overexposed Detection**: Detects overexposed or mostly white images
- **Multiprocessing**: Fast processing using all CPU cores
- **Flexible Output**: Organize flagged images into separate folders
- **Progress Tracking**: Visual progress bar during processing

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the script with one of three detection modes:

```bash
# Detect and move blurry images
python image-helper.py blurry

# Detect and move mostly black images
python image-helper.py black

# Detect and move overexposed/white images
python image-helper.py white
```

### Advanced Options

**Custom input folder:**
```bash
python image-helper.py blurry --input /path/to/your/photos
```

**Custom output folder:**
```bash
python image-helper.py blurry --output /path/to/blurry/photos
```

**Combined example:**
```bash
python image-helper.py white --input /Users/joe/Photos --output /Users/joe/overexposed
```

### Command Line Arguments

- `check_type` (required): Choose from `black`, `white`, or `blurry`
- `--input, -i`: Input folder containing images (default: configured in script)
- `--output, -o`: Output folder for flagged images (default: creates folder named after check type)

## Configuration

Edit the configuration section at the top of `image-helper.py` to adjust detection thresholds:

```python
# --- CONFIGURATION ---
input_folder = "/Users/joe/Code/photo-organizer/source"  # Default input folder
output_folder_base = "/Users/joe/Code/photo-organizer"   # Base folder for outputs
black_threshold = 30        # Pixel value below which it's considered "black"
black_ratio_cutoff = 0.9    # 90% black = flag the image
white_threshold = 225       # Pixel value above this is "white"
white_ratio_cutoff = 0.9    # Flag if 90% of pixels are white
resize_dims = (192, 108)    # Downscale for speed; preserves aspect ratio
blur_threshold = 300.0      # Lower = more blurry; tweak as needed
```

### Threshold Tuning

- **Blur threshold**: Lower values detect more images as blurry (try 100-500)
- **Black/White ratios**: Adjust between 0.8-0.95 for sensitivity
- **Pixel thresholds**: Fine-tune what constitutes "black" or "white" pixels

## Supported File Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif)

## How It Works

1. **Scans** the input folder for supported image files
2. **Processes** images in parallel using all CPU cores
3. **Applies** the selected detection algorithm:
   - **Blur**: Uses Laplacian variance to measure image sharpness
   - **Black**: Calculates percentage of dark pixels
   - **White**: Calculates percentage of bright pixels
4. **Moves** flagged images to the output folder
5. **Reports** results and any errors

## Output Structure

By default, the script creates folders based on the detection type:

```
photo-organizer/
├── source/          # Your original photos
├── blurry/          # Blurry images moved here
├── black/           # Mostly black images moved here
├── white/           # Overexposed images moved here
└── image-helper.py  # The script
```

## Tips

- **Start with a small test folder** to calibrate thresholds
- **Review flagged images** before permanent deletion
- **Run different detection modes** to catch various issues
- **Backup your photos** before running the script
- **Adjust thresholds** based on your specific photo collection

## Example Workflow

```bash
# 1. Check for blurry images
python image-helper.py blurry --input /Users/joe/Photos

# 2. Review the blurry/ folder, adjust blur_threshold if needed

# 3. Check for overexposed images
python image-helper.py white --input /Users/joe/Photos

# 4. Check for blank/black images
python image-helper.py black --input /Users/joe/Photos
```

## Troubleshooting

- **No images found**: Check that your input folder path is correct
- **Script runs slowly**: Reduce `resize_dims` or limit CPU cores
- **Too many false positives**: Increase thresholds (blur_threshold, ratio_cutoffs)
- **Missing images**: Decrease thresholds to be more sensitive

## License

This project is open source. Feel free to modify and distribute.
