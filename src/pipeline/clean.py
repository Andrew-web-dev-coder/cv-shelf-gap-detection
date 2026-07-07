import cv2
import numpy as np

def clean_mask(mask):
    
    if mask is None:
        return None
    
    
    kernel = np.ones((5, 5), np.uint8)
    
    
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
   
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
   
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    min_area = 300  
    result = np.zeros_like(cleaned)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255
    
  
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return result