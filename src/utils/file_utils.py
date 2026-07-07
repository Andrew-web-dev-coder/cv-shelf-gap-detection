# src/utils/file_utils.py
import os
import glob
from pathlib import Path

def ensure_directories():
    dirs = [
        'data/raw',
        'data/processed/01_original',
        'data/processed/02_enhanced',
        'data/processed/03_masks',
        'data/processed/04_cleaned',
        'data/processed/05_detection',
        'data/processed/06_final',
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_input_images(input_dir='data/raw'):

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    return list(set(image_files))  

def get_output_paths(base_name, output_dir='data/processed'):

    return {
        'original': os.path.join(output_dir, '01_original', f'{base_name}_original.jpg'),
        'enhanced': os.path.join(output_dir, '02_enhanced', f'{base_name}_enhanced.jpg'),
        'mask': os.path.join(output_dir, '03_masks', f'{base_name}_mask.jpg'),
        'cleaned': os.path.join(output_dir, '04_cleaned', f'{base_name}_cleaned.jpg'),
        'detection': os.path.join(output_dir, '05_detection', f'{base_name}_detection.jpg'),
        'final': os.path.join(output_dir, '06_final', f'{base_name}_final.jpg'),
        'collage': os.path.join(output_dir, f'{base_name}_collage.jpg'),
    }