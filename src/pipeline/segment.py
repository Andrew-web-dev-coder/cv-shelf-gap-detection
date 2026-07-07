# src/pipeline/segment.py
import cv2
import numpy as np

def segment_bookshelves(image):
 
    if image is None:
        return None
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    return mask