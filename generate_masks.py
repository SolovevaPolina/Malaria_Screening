import os
import cv2
import numpy as np
import glob
from tqdm.auto import tqdm

# Define relative path to the dataset based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(script_dir, 'NIH-NLM-ThinBloodSmearsPf', 'Polygon Set')

def parse_and_create_mask(txt_path):
    """
    Parses the annotation text file and generates a binary segmentation mask.
    Returns: numpy array (height, width) where 255 is cell, 0 is background.
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None

    # Count, Width, Height
    header = lines[0].strip().split(',')
    img_width = int(header[1])
    img_height = int(header[2])
    
    # Initialize black mask (single channel)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Parse cell annotations
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 6: continue 
        
        # Coordinates start at index 5
        coords_raw = parts[5:]
        coords_float = [float(x) for x in coords_raw]
        
        # Reshape for OpenCV: (N, 2) array of points
        pts = np.array(coords_float, np.int32).reshape((-1, 2))
        
        cv2.fillPoly(mask, [pts], color=255)
        
    return mask

def main():
    search_path = os.path.join(dataset_root, "**", "GT", "*.txt")
    txt_files = glob.glob(search_path, recursive=True)
    
    print(f"Found annotation files: {len(txt_files)}")
    
    for txt_path in tqdm(txt_files, desc="Generating masks"):
        gt_folder = os.path.dirname(txt_path)
        patient_folder = os.path.dirname(gt_folder)
        
        # Create 'Masks' directory if it doesn't exist
        mask_folder = os.path.join(patient_folder, "Masks")
        os.makedirs(mask_folder, exist_ok=True)
        
        filename = os.path.basename(txt_path)
        filename_png = filename.replace('.txt', '.png')
        save_path = os.path.join(mask_folder, filename_png)
            
        # Generate and save mask
        mask = parse_and_create_mask(txt_path)
        
        if mask is not None:
            cv2.imwrite(save_path, mask)

    print("\nMasks saved")

if __name__ == "__main__":
    main()