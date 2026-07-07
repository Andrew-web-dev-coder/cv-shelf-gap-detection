# src/pipeline/detect.py
from ultralytics import YOLO
import numpy as np
import cv2

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO('yolov8n.pt')
    return _model


def detect_books_and_gaps(image, mask=None):
    """
    Planogram-Aware Detection: Finds books and detects REAL gaps
    (places where books are missing based on expected shelf structure)
    """
    if image is None:
        return [], []

    model = get_model()
    height, width = image.shape[:2]

    # ========== STEP 0: CHECK IMAGE BRIGHTNESS ==========
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
   
    conf_threshold = 0.10 if avg_brightness < 100 else 0.15
    
    # ========== STEP 1: DETECT BOOKS WITH YOLO ==========
    results = model(image, conf=conf_threshold, verbose=False)

    books = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]

            
            min_conf = 0.08 if avg_brightness < 100 else 0.1
            if (cls_name == 'book' or conf > 0.25) and conf > min_conf:
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                if w > 15 and h > 40 and (h / w) > 0.8:
                    books.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'confidence': conf
                    })

    # ========== FALLBACK: HOG DETECTION IF YOLO FINDS NOTHING ==========
    if len(books) == 0:
        print("   ⚠️ YOLO found no books, trying HOG fallback...")
        books = detect_books_hog(image)
        if len(books) > 0:
            print(f"   ✅ HOG found {len(books)} books")

    
    books = sorted(books, key=lambda b: b['x'])

    # ========== STEP 2: FIND REAL GAPS (Planogram-Aware) ==========
    real_gaps = []

    if len(books) >= 2:
        
        widths = [b['w'] for b in books]
        heights = [b['h'] for b in books]
        median_width = np.median(widths) if widths else 0
        median_height = np.median(heights) if heights else 0

       
        if median_width < 30:
            median_width = np.mean(widths) if widths else 0

       
        shelf_left = min([b['x'] for b in books])
        shelf_right = max([b['x'] + b['w'] for b in books])
        shelf_width = shelf_right - shelf_left

        
        expected_spacing = median_width * 1.1
        expected_books = int(shelf_width / expected_spacing) + 1

      
       
        for i in range(len(books) - 1):
            b1 = books[i]
            b2 = books[i + 1]

            gap_start = b1['x'] + b1['w']
            gap_end = b2['x']
            gap_width = gap_end - gap_start

            
            overlap = min(b1['y'] + b1['h'], b2['y'] + b2['h']) - max(b1['y'], b2['y'])

            
            if gap_width > median_width * 0.3 and overlap > median_height * 0.15:
                missing_books = max(1, int(gap_width / (median_width * 1.1)))

                
                if gap_width >= median_width * 1.5:
                    gap_type = 'large'
                    severity = 'critical'
                    action = 'RESTOCK IMMEDIATELY'
                elif gap_width >= median_width * 0.8:
                    gap_type = 'medium'
                    severity = 'moderate'
                    action = 'CHECK INVENTORY'
                elif gap_width >= median_width * 0.4:
                    gap_type = 'small'
                    severity = 'minor'
                    action = 'MONITOR'
                else:
                    gap_type = 'tiny'
                    severity = 'info'
                    action = 'OBSERVE'

                real_gaps.append({
                    'x': gap_start,
                    'y': max(b1['y'], b2['y']),
                    'width': gap_width,
                    'height': min(overlap, median_height * 1.2),
                    'left_book': i,
                    'right_book': i + 1,
                    'gap_width': gap_width,
                    'gap_type': gap_type,
                    'severity': severity,
                    'action': action,
                    'missing_books': missing_books,
                    'expected_books_between': missing_books
                })

       
        if len(books) > 0:
            first_book = books[0]
            left_gap = first_book['x'] - shelf_left
            if left_gap > median_width * 0.4:
                missing_books = max(1, int(left_gap / (median_width * 1.1)))
                real_gaps.append({
                    'x': shelf_left,
                    'y': first_book['y'],
                    'width': left_gap,
                    'height': first_book['h'],
                    'left_book': -1,
                    'right_book': 0,
                    'gap_width': left_gap,
                    'gap_type': 'edge_left',
                    'severity': 'critical' if left_gap > median_width * 1.5 else 'moderate',
                    'action': 'CHECK INVENTORY' if left_gap > median_width * 1.5 else 'MONITOR',
                    'missing_books': missing_books,
                    'expected_books_between': missing_books
                })

           
            last_book = books[-1]
            right_gap = shelf_right - (last_book['x'] + last_book['w'])
            if right_gap > median_width * 0.4:
                missing_books = max(1, int(right_gap / (median_width * 1.1)))
                real_gaps.append({
                    'x': last_book['x'] + last_book['w'],
                    'y': last_book['y'],
                    'width': right_gap,
                    'height': last_book['h'],
                    'left_book': len(books) - 1,
                    'right_book': -1,
                    'gap_width': right_gap,
                    'gap_type': 'edge_right',
                    'severity': 'critical' if right_gap > median_width * 1.5 else 'moderate',
                    'action': 'CHECK INVENTORY' if right_gap > median_width * 1.5 else 'MONITOR',
                    'missing_books': missing_books,
                    'expected_books_between': missing_books
                })

        # ========== METHOD 3: Compare actual vs expected book count ==========
        if len(books) < expected_books * 0.5:
            missing_total = expected_books - len(books)
            found_missing = sum([g['missing_books'] for g in real_gaps])
            if found_missing < missing_total:
             
                if real_gaps:
                 
                    largest = max(real_gaps, key=lambda g: g['gap_width'])
                    largest['missing_books'] += (missing_total - found_missing)
                    largest['expected_books_between'] = largest['missing_books']
                else:
                    
                    real_gaps.append({
                        'x': shelf_left + median_width,
                        'y': books[0]['y'] if books else 0,
                        'width': median_width * 2,
                        'height': median_height,
                        'left_book': -1,
                        'right_book': -1,
                        'gap_width': median_width * 2,
                        'gap_type': 'undetected',
                        'severity': 'critical',
                        'action': 'REVIEW IMAGE',
                        'missing_books': missing_total,
                        'expected_books_between': missing_total
                    })

    # ========== STEP 3: CALCULATE EXPECTED VS ACTUAL ==========
    total_expected = len(books)
    if len(books) > 1:
        total_missing = sum([g['missing_books'] for g in real_gaps])
        total_expected = len(books) + total_missing

 
    print(f"   📚 Books found: {len(books)}")
    print(f"   📏 Real gaps detected: {len(real_gaps)}")
    if real_gaps:
        gap_details = [f"{g['gap_width']:.0f}px ({g['gap_type']})" for g in real_gaps]
        print(f"   📏 Gap sizes: {gap_details}")
        missing_total = sum([g['missing_books'] for g in real_gaps])
        print(f"   ❗ Missing books: ~{missing_total}")

    return books, real_gaps


def detect_books_hog(image):
    """
    Fallback method: Detect books using HOG (Histogram of Oriented Gradients)
    Used when YOLO fails to find any books
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        gray = cv2.equalizeHist(gray)
        
        
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        
        rects, weights = hog.detectMultiScale(
            gray, 
            winStride=(4, 4), 
            padding=(8, 8), 
            scale=1.05,
            hitThreshold=0.8
        )
        
        books = []
        for (x, y, w, h) in rects:
           
            aspect_ratio = h / w if w > 0 else 0
            if w > 15 and h > 40 and aspect_ratio > 1.0 and w < 150 and h < 500:
                
                is_duplicate = False
                for existing in books:
                    overlap_x = max(0, min(x + w, existing['x'] + existing['w']) - max(x, existing['x']))
                    overlap_y = max(0, min(y + h, existing['y'] + existing['h']) - max(y, existing['y']))
                    overlap_area = overlap_x * overlap_y
                    if overlap_area > (w * h) * 0.3:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    
                    conf = min(0.8, 0.3 + (weights[0] if len(weights) > 0 else 0.5))
                    books.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'confidence': conf
                    })
        
       
        books = sorted(books, key=lambda b: b['confidence'], reverse=True)
        
        
        return books[:50]
        
    except Exception as e:
        print(f"   ⚠️ HOG fallback failed: {e}")
        return []