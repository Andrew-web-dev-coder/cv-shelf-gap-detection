# src/main.py
import cv2
import os
import time
import numpy as np
from pipeline.enhance import enhance_image
from pipeline.segment import segment_bookshelves
from pipeline.clean import clean_mask
from pipeline.detect import detect_books_and_gaps
from pipeline.decide import decide
from utils.visualize import save_results
from utils.file_utils import ensure_directories, get_input_images, get_output_paths

def process_image(image_path, output_base='data/processed'):
    
    try:
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load: {image_path}")
            return None
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n📷 Processing: {os.path.basename(image_path)}")
        print(f"   Shape: {image.shape}")
        
        # 1. Enhance
        enhanced = enhance_image(image)
        
        # 2. Segment (упрощен для YOLO)
        mask = segment_bookshelves(enhanced)
        
        # 3. Clean
        cleaned = clean_mask(mask)
        
        # 4. Detect (YOLO)
        books, gaps = detect_books_and_gaps(enhanced, cleaned)
        
        # 5. Decide
        result = decide(books, gaps, image.shape)
        print(f"   📊 Status: {result['status']}")
        print(f"   📝 Decision: {result['decision']}")
        
        
        paths = get_output_paths(base_name, output_base)
        save_results(image, enhanced, mask, cleaned, books, gaps, result, paths)
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
   
    print("=" * 60)
    print("📚 SHELF GAP DETECTION SYSTEM")
    print("=" * 60)
    
    
    ensure_directories()
    
   
    images = get_input_images('data/raw')
    
    if not images:
        print("❌ No images found in data/raw/")
        print("📁 Please add images to data/raw/ folder")
        return
    
    print(f"📷 Found {len(images)} images")
    print("-" * 60)
    
  
    results = []
    start_time = time.time()
    
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]", end=" ")
        result = process_image(img_path)
        if result:
            results.append({
                'image': os.path.basename(img_path),
                'result': result
            })
    
  
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Processed: {len(results)} images")
    print(f"⏱️ Time: {elapsed:.2f}s")
    print("\nResults:")
    
    for r in results:
        print(f"  📄 {r['image']}")
        print(f"     Status: {r['result']['status']}")
        print(f"     Books: {r['result']['total_books']}, Gaps: {r['result']['gap_count']}")
    
    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    main()