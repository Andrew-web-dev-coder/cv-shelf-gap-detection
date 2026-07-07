# src/utils/visualize.py
import cv2
import numpy as np
import os

def save_results(image, enhanced, mask, cleaned, books, gaps, result, paths):
   
   
    for path in paths.values():
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    
   
    cv2.imwrite(paths['original'], image)
    
    
    if enhanced is not None:
        cv2.imwrite(paths['enhanced'], enhanced)
    
    
    if mask is not None:
        cv2.imwrite(paths['mask'], mask)
    
  
    if cleaned is not None:
        cv2.imwrite(paths['cleaned'], cleaned)
    
   
    det_img = image.copy()
    
   
    for i, book in enumerate(books):
        x, y, w, h = book['x'], book['y'], book['w'], book['h']
        cv2.rectangle(det_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(det_img, f"Book {i+1}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        conf = book.get('confidence', 0)
        if conf > 0:
            cv2.putText(det_img, f"{conf:.2f}", (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
   
    for gap in gaps:
        x, y, w, h = gap['x'], gap['y'], gap['width'], gap['height']
        
        
        cv2.rectangle(det_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        
        gap_text = f"GAP {gap['width']}px"
        cv2.putText(det_img, gap_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
       
        if gap['width'] < 30:
            cv2.putText(det_img, "⬆", (x + w//2 - 5, y + h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
    info_text = f"Books: {len(books)} | Gaps: {len(gaps)}"
    cv2.putText(det_img, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(paths['detection'], det_img)
    
    final_img = image.copy()
    
    status_colors = {
        'OK': (0, 255, 0),
        'RESTOCK_NEEDED': (0, 0, 255),
        'CHECK_INVENTORY': (0, 165, 255),
        'MINOR_GAPS': (255, 255, 0),
        'TINY_GAPS': (255, 200, 0),
        'LOW_BOOKS': (0, 0, 255),
        'ERROR': (0, 0, 255)
    }
    
    color = status_colors.get(result['status'], (255, 255, 255))
    
    y = 30
    cv2.putText(final_img, f"Status: {result['status']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y += 35
    cv2.putText(final_img, f"Books: {result['total_books']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30
    cv2.putText(final_img, f"Gaps: {result['gap_count']}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if result['gap_count'] > 0:
        y += 30
        gap_info = f"Large: {result.get('large_gaps', 0)}, Medium: {result.get('medium_gaps', 0)}, Small: {result.get('small_gaps', 0)}"
        cv2.putText(final_img, gap_info, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
    h, w = final_img.shape[:2]
    cv2.putText(final_img, result['decision'], 
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(paths['final'], final_img)
    
    
    create_collage(image, enhanced, mask, cleaned, det_img, final_img, paths['collage'])

def create_collage(original, enhanced, mask, cleaned, detection, final, output_path):
    
    h, w = original.shape[:2]
    new_h, new_w = h // 2, w // 2
    
   
    images = []
    
    
    images.append(cv2.resize(original, (new_w, new_h)))
    
   
    if enhanced is not None:
        images.append(cv2.resize(enhanced, (new_w, new_h)))
    else:
        images.append(np.zeros((new_h, new_w, 3), dtype=np.uint8))
    
    
    images.append(cv2.resize(detection, (new_w, new_h)))
    
    
    images.append(cv2.resize(final, (new_w, new_h)))
    
  
    if mask is not None:
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        images.append(cv2.resize(mask_color, (new_w, new_h)))
    else:
        images.append(np.zeros((new_h, new_w, 3), dtype=np.uint8))
    
   
    if cleaned is not None:
        cleaned_color = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        images.append(cv2.resize(cleaned_color, (new_w, new_h)))
    else:
        images.append(np.zeros((new_h, new_w, 3), dtype=np.uint8))
    
    
    top = np.hstack(images[:3])
    bottom = np.hstack(images[3:6])
    collage = np.vstack([top, bottom])
    
    titles = ['Original', 'Enhanced', 'Detection', 'Final', 'Mask', 'Cleaned']
    for i, title in enumerate(titles):
        col = i % 3
        row = i // 3
        x = col * new_w + 10
        y = row * new_h + 30
        cv2.putText(collage, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, collage)