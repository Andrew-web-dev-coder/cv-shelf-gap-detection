# src/pipeline/decide.py
import numpy as np

def decide(books, gaps, image_shape=None):
    
    total_books = len(books)
    total_gaps = len(gaps)

    # ========== CASE 1: NO BOOKS DETECTED ==========
    if total_books == 0:
        return {
            'decision': 'ERROR: No books detected. Check image quality.',
            'status': 'ERROR',
            'gap_count': 0,
            'total_books': 0,
            'large_gaps': 0,
            'medium_gaps': 0,
            'small_gaps': 0,
            'tiny_gaps': 0,
            'gap_sizes': [],
            'missing_books_total': 0,
            'expected_books': 0,
            'shelf_status': 'unknown',
            'message': 'No books found',
            'detailed_gaps': []
        }

    # ========== CALCULATE STATISTICS ==========
    
    critical_gaps = [g for g in gaps if g.get('severity') == 'critical']
    moderate_gaps = [g for g in gaps if g.get('severity') == 'moderate']
    minor_gaps = [g for g in gaps if g.get('severity') == 'minor']

    
    large_gaps = [g for g in gaps if g.get('gap_type') in ['large', 'edge_left', 'edge_right']]
    medium_gaps = [g for g in gaps if g.get('gap_type') == 'medium']
    small_gaps = [g for g in gaps if g.get('gap_type') in ['small', 'undetected']]

    
    missing_books_total = sum([g.get('missing_books', 1) for g in gaps])
    expected_books = total_books + missing_books_total

    # ========== DETERMINE SHELF STATUS ==========
    if total_books < 3:
        status = 'LOW_BOOKS'
        decision = f'WARNING: Only {total_books} books detected. Shelf appears mostly empty.'

    elif total_gaps == 0:
        status = 'OK'
        decision = 'OK: Shelf is complete. No gaps detected.'

    elif critical_gaps:
        status = 'RESTOCK_NEEDED'
        reasons = []
        for g in critical_gaps:
            if g.get('gap_type') == 'edge_left':
                reasons.append(f"large gap at left edge ({g['gap_width']:.0f}px)")
            elif g.get('gap_type') == 'edge_right':
                reasons.append(f"large gap at right edge ({g['gap_width']:.0f}px)")
            else:
                reasons.append(f"gap between books ({g['gap_width']:.0f}px)")
        decision = f'RESTOCK NEEDED: {len(critical_gaps)} critical gaps found. Missing ~{missing_books_total} books. {", ".join(reasons[:2])}.'

    elif moderate_gaps:
        status = 'CHECK_INVENTORY'
        decision = f'CHECK INVENTORY: {len(moderate_gaps)} moderate gaps found. Missing ~{missing_books_total} books.'

    elif large_gaps:
        status = 'LARGE_GAPS'
        decision = f'WARNING: {len(large_gaps)} large gaps found. Missing ~{missing_books_total} books.'

    elif medium_gaps:
        status = 'MEDIUM_GAPS'
        decision = f'MEDIUM GAPS: {len(medium_gaps)} medium gaps found. Missing ~{missing_books_total} books.'

    elif small_gaps:
        status = 'MINOR_GAPS'
        decision = f'MINOR GAPS: {len(small_gaps)} small gaps found. Missing ~{missing_books_total} books.'

    else:
        status = 'TINY_GAPS'
        decision = f'TINY GAPS: {len([g for g in gaps if g.get("gap_type") == "tiny"])} very small gaps found.'

    # ========== BUILD DETAILED RESPONSE ==========
    return {
        'decision': decision,
        'status': status,
        'gap_count': total_gaps,
        'total_books': total_books,
        'expected_books': expected_books,
        'missing_books_total': missing_books_total,
        'large_gaps': len(large_gaps),
        'medium_gaps': len(medium_gaps),
        'small_gaps': len(small_gaps),
        'critical_gaps': len(critical_gaps),
        'moderate_gaps': len(moderate_gaps),
        'minor_gaps': len(minor_gaps),
        'gap_sizes': [g['gap_width'] for g in gaps],
        'gap_details': [
            {
                'width': g['gap_width'],
                'type': g.get('gap_type', 'unknown'),
                'severity': g.get('severity', 'unknown'),
                'missing_books': g.get('missing_books', 1),
                'action': g.get('action', 'REVIEW')
            }
            for g in gaps
        ],
        'detailed_gaps': gaps,
        'shelf_completeness': f"{total_books}/{expected_books} books ({int(total_books/expected_books*100) if expected_books > 0 else 0}%)",
        'message': f'Found {total_books} books, missing ~{missing_books_total} books. {total_gaps} gaps detected.'
    }