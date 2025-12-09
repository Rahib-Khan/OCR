"""
Streamlined Batch OCR System for Historical Documents
Performs preprocessing, OCR, visualization, and CSV export
Optimized for 400 DPI table extraction
"""

import cv2
import numpy as np
import pytesseract
import pandas as pd
from typing import List, Dict, Tuple
import os
import glob
import re


class BatchOCRExtractor:
    """
    Streamlined OCR extraction system for batch processing.
    Optimized for high-resolution scans of historical data tables.
    """
    
    def __init__(self, image_path: str, dpi: int = 400):
        """
        Initialize the OCR extractor with an image file.
        
        Args:
            image_path: Path to the input image file
            dpi: DPI of the input image (affects preprocessing parameters)
        """
        self.image_path = image_path
        self.dpi = dpi
        self.original = None
        self.processed = None
        self.warped = None
        self.results = []
        self.used_fallback_contour = False
        self.column_lines = []
        
        # Adjust processing parameters based on DPI
        self.scale_factor = dpi / 300.0  # Scale relative to 300 DPI baseline
        
    def load_image(self) -> np.ndarray:
        """Load the image from file."""
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")
        return self.original
    
    def find_document_contour(self, image: np.ndarray) -> np.ndarray:
        """
        Find the largest rectangular contour in the image (document boundary).
        Uses conservative approach to avoid detecting table columns as document edges.
        
        Args:
            image: Input image
            
        Returns:
            Array of 4 corner points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        image_area = h * w
        
        # Adjust blur kernel based on DPI
        kernel_size = max(5, int(5 * self.scale_factor))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Adaptive edge detection thresholds for high DPI
        low_threshold = int(50 * self.scale_factor)
        high_threshold = int(150 * self.scale_factor)
        edged = cv2.Canny(blurred, low_threshold, high_threshold)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        # Look for valid document contours with area constraints
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Calculate contour area
                contour_area = cv2.contourArea(approx)
                area_ratio = contour_area / image_area
                
                # Only accept contours that are at least 50% of image area
                # This prevents small table columns from being detected as document boundary
                if area_ratio > 0.5:
                    # Additional check: ensure it's roughly rectangular (not too skewed)
                    rect = self.order_points(approx)
                    width1 = np.linalg.norm(rect[1] - rect[0])
                    width2 = np.linalg.norm(rect[2] - rect[3])
                    height1 = np.linalg.norm(rect[3] - rect[0])
                    height2 = np.linalg.norm(rect[2] - rect[1])
                    
                    # Check if opposite sides are similar length (within 20%)
                    width_ratio = min(width1, width2) / max(width1, width2)
                    height_ratio = min(height1, height2) / max(height1, height2)
                    
                    if width_ratio > 0.8 and height_ratio > 0.8:
                        self.used_fallback_contour = False
                        return approx
        
        # Fallback to image corners if no valid contour found
        print("  [Warning] No valid document contour found, using full image")
        self.used_fallback_contour = True
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points: top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        pts = pts.reshape(4, 2)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def perspective_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to get top-down view.
        """
        rect = self.order_points(pts)
        tl, tr, br, bl = rect
        
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def detect_table_structure(self, image: np.ndarray) -> List[int]:
        """
        Detect vertical column lines in the table to aid OCR.
        
        Args:
            image: Preprocessed binary image
            
        Returns:
            List of x-coordinates for detected vertical lines
        """
        # Create a copy for line detection
        lines_img = image.copy()
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image.shape[0] * 0.1)))
        detect_vertical = cv2.morphologyEx(~lines_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Find contours of vertical lines
        contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract x-coordinates of vertical lines
        vertical_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter: must be tall (at least 20% of image height) and thin
            if h > image.shape[0] * 0.2 and w < 10:
                vertical_lines.append(x)
        
        # Sort lines from left to right
        vertical_lines = sorted(set(vertical_lines))
        
        return vertical_lines
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing optimized for 400 DPI historical documents.
        Enhanced to better detect table structure.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For 400 DPI images, use optimized denoising
        # Bilateral filter with larger diameter for 400 DPI
        d = 11  # Optimized for 400 DPI
        denoised = cv2.bilateralFilter(gray, d, 80, 80)
        
        # Enhance contrast using CLAHE (helps with faded text)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Enhance table lines before thresholding
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel_line)
        
        # Combine original with gradient to emphasize lines
        combined = cv2.addWeighted(enhanced, 0.8, gradient, 0.2, 0)
        
        # Adaptive thresholding optimized for 400 DPI
        block_size = 15  # Optimized for 400 DPI tables
        thresh = cv2.adaptiveThreshold(
            combined, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 2
        )
        
        # Clean up with morphological closing (removes small noise)
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Detect table structure
        self.column_lines = self.detect_table_structure(morph)
        if self.column_lines:
            print(f"  Detected {len(self.column_lines)} column lines at x-positions: {self.column_lines[:5]}...")
        
        # Add padding optimized for 400 DPI
        padding = 60  # Slightly larger for 400 DPI
        padded = cv2.copyMakeBorder(
            morph, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=255
        )
        
        return padded
    
    def get_column_index(self, x_position: int) -> int:
        """
        Determine which column a text element belongs to based on x-position.
        
        Args:
            x_position: X coordinate of the text element
            
        Returns:
            Column index (0 = first column, 1 = second column, etc.)
        """
        if not self.column_lines:
            return 0
        
        # Adjust for padding
        padding = int(50 * self.scale_factor)
        adjusted_x = x_position - padding
        
        # Find which column the x position falls into
        for i, line_x in enumerate(self.column_lines):
            if adjusted_x < line_x:
                return i
        
        return len(self.column_lines)  # Last column
    
    def clip_box_to_column(self, left: int, width: int, col_idx: int) -> Tuple[int, int]:
        """
        Clip bounding box to stay within column boundaries.
        
        Args:
            left: Original left x-coordinate
            width: Original width
            col_idx: Column index
            
        Returns:
            Tuple of (clipped_left, clipped_width)
        """
        if not self.column_lines or col_idx >= len(self.column_lines):
            return left, width
        
        padding = int(50 * self.scale_factor)
        
        # Get column boundaries
        if col_idx == 0:
            col_start = 0
            col_end = self.column_lines[0] + padding
        elif col_idx < len(self.column_lines):
            col_start = self.column_lines[col_idx - 1] + padding
            col_end = self.column_lines[col_idx] + padding
        else:
            col_start = self.column_lines[-1] + padding
            col_end = float('inf')
        
        # Clip left boundary
        clipped_left = max(left, col_start)
        
        # Clip right boundary
        right = left + width
        clipped_right = min(right, col_end) if col_end != float('inf') else right
        
        # Calculate new width
        clipped_width = max(0, clipped_right - clipped_left)
        
        return clipped_left, clipped_width
    
    def perform_ocr(self, image: np.ndarray, page_num: int = 1, 
                    multi_pass: bool = True) -> List[Dict]:
        """
        Perform OCR with bounding box extraction.
        Uses multi-pass strategy and applies column-aware character restrictions.
        Clips boxes to column boundaries to prevent cross-column errors.
        
        Args:
            image: Preprocessed image
            page_num: Page number for metadata
            multi_pass: If True, run multiple OCR passes with different configs
            
        Returns:
            List of dictionaries containing OCR results
        """
        results = []
        
        # Pass 1: First column - allow alphanumeric + common punctuation
        config_1 = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        ocr_data_1 = pytesseract.image_to_data(
            image, 
            config=config_1,
            output_type=pytesseract.Output.DICT
        )
        
        n_boxes = len(ocr_data_1['text'])
        for i in range(n_boxes):
            if int(ocr_data_1['conf'][i]) > 0:
                text = ocr_data_1['text'][i].strip()
                if text:
                    left = ocr_data_1['left'][i]
                    top = ocr_data_1['top'][i]
                    width = ocr_data_1['width'][i]
                    col_idx = self.get_column_index(left)
                    
                    # Clip box to column boundaries
                    clipped_left, clipped_width = self.clip_box_to_column(left, width, col_idx)
                    
                    # Skip if box is completely clipped
                    if clipped_width <= 0:
                        continue
                    
                    results.append({
                        'level': ocr_data_1['level'][i],
                        'page_num': page_num,
                        'block_num': ocr_data_1['block_num'][i],
                        'par_num': ocr_data_1['par_num'][i],
                        'line_num': ocr_data_1['line_num'][i],
                        'word_num': ocr_data_1['word_num'][i],
                        'left': clipped_left,
                        'top': top,
                        'width': clipped_width,
                        'height': ocr_data_1['height'][i],
                        'conf': int(ocr_data_1['conf'][i]),
                        'text': text,
                        'column_index': col_idx,
                        'pass': 1
                    })
        
        if not multi_pass:
            return results
        
        # Pass 2: Focus on numeric data (for data columns after first column)
        config_2 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        try:
            ocr_data_2 = pytesseract.image_to_data(
                image, 
                config=config_2,
                output_type=pytesseract.Output.DICT
            )
            
            # Track existing bounding boxes to avoid duplicates
            existing_boxes = {(r['left'], r['top'], r['width'], r['height']) 
                            for r in results}
            
            n_boxes_2 = len(ocr_data_2['text'])
            for i in range(n_boxes_2):
                if int(ocr_data_2['conf'][i]) > 0:
                    text = ocr_data_2['text'][i].strip()
                    if text and text.replace('.', '').replace(',', '').replace('-', '').isdigit():
                        left = ocr_data_2['left'][i]
                        width = ocr_data_2['width'][i]
                        col_idx = self.get_column_index(left)
                        
                        # Clip box to column boundaries
                        clipped_left, clipped_width = self.clip_box_to_column(left, width, col_idx)
                        
                        # Skip if box is completely clipped
                        if clipped_width <= 0:
                            continue
                        
                        box = (clipped_left, ocr_data_2['top'][i], 
                               clipped_width, ocr_data_2['height'][i])
                        
                        # Only add if not already found in pass 1
                        if box not in existing_boxes:
                            results.append({
                                'level': ocr_data_2['level'][i],
                                'page_num': page_num,
                                'block_num': ocr_data_2['block_num'][i],
                                'par_num': ocr_data_2['par_num'][i],
                                'line_num': ocr_data_2['line_num'][i],
                                'word_num': ocr_data_2['word_num'][i],
                                'left': clipped_left,
                                'top': ocr_data_2['top'][i],
                                'width': clipped_width,
                                'height': ocr_data_2['height'][i],
                                'conf': int(ocr_data_2['conf'][i]),
                                'text': text,
                                'column_index': col_idx,
                                'pass': 2
                            })
        except Exception as e:
            print(f"  [Warning] Pass 2 failed: {e}")
        
        # Sort results by position (top to bottom, left to right) and reassign line numbers
        results = self.reassign_line_numbers(results)
        
        return results
    
    def reassign_line_numbers(self, results: List[Dict]) -> List[Dict]:
        """
        Reassign line numbers based on vertical position to group text on same line.
        
        Args:
            results: List of OCR results
            
        Returns:
            Updated results with corrected line numbers
        """
        if not results:
            return results
        
        # Sort by top position
        sorted_results = sorted(results, key=lambda x: (x['top'], x['left']))
        
        # Group by similar vertical positions (within 10 pixels)
        line_threshold = 10
        current_line = 1
        last_top = sorted_results[0]['top']
        
        for result in sorted_results:
            if abs(result['top'] - last_top) > line_threshold:
                current_line += 1
                last_top = result['top']
            
            result['line_num'] = current_line
        
        return sorted_results
    
    def visualize_ocr(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and text on the image.
        Uses clear color coding: high confidence = solid green, low confidence = red/orange.
        Draws column lines for reference.
        """
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Scale visualization parameters for 400 DPI
        line_thickness = max(2, int(2.5 * self.scale_factor))
        font_scale = max(0.4, 0.5 * self.scale_factor)
        
        # Draw detected column lines (thin blue lines)
        padding = int(50 * self.scale_factor)
        for line_x in self.column_lines:
            adjusted_x = line_x + padding
            cv2.line(vis_image, (adjusted_x, 0), (adjusted_x, vis_image.shape[0]), 
                    (200, 150, 0), max(1, line_thickness // 3))
        
        # Draw bounding boxes with confidence-based colors
        for result in results:
            x, y, w, h = result['left'], result['top'], result['width'], result['height']
            text = result['text']
            conf = result['conf']
            
            # Color based on confidence:
            # High (>70): Bright Green
            # Medium (50-70): Yellow/Orange
            # Low (<50): Red
            if conf > 70:
                color = (0, 255, 0)  # Bright green
            elif conf > 50:
                color = (0, 200, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, line_thickness)
            
            # Draw text label above box
            display_text = text[:15] + '...' if len(text) > 15 else text
            
            # Add background rectangle for text visibility
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale * 0.8, 1)[0]
            cv2.rectangle(vis_image, 
                         (x, y - text_size[1] - 8), 
                         (x + text_size[0] + 4, y - 2),
                         (255, 255, 255), -1)
            
            cv2.putText(vis_image, display_text, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)
        
        return vis_image
    
    def save_column_structure(self, output_dir: str, page_num: int):
        """
        Save detected column structure to a file for later reconstruction.
        
        Args:
            output_dir: Directory to save column info
            page_num: Page number
        """
        if not self.column_lines:
            return
        
        basename = os.path.splitext(os.path.basename(self.image_path))[0]
        column_file = os.path.join(output_dir, f"{basename}_columns.txt")
        
        with open(column_file, 'w') as f:
            f.write(f"# Column structure for page {page_num}\n")
            f.write(f"# Image: {self.image_path}\n")
            f.write(f"# Number of columns: {len(self.column_lines) + 1}\n")
            f.write(f"# Column line x-positions (in pixels):\n")
            for i, x_pos in enumerate(self.column_lines, 1):
                f.write(f"line_{i}: {x_pos}\n")
    
    def process(self, page_num: int = 1, output_dir: str = "Processed",
                multi_pass: bool = True) -> Dict:
        """
        Execute the complete OCR pipeline.
        
        Args:
            page_num: Page number for metadata
            output_dir: Directory to save processed images
            multi_pass: Use multi-pass OCR to catch single digits
        
        Returns:
            Dictionary with paths to output files and results
        """
        print(f"\n[Page {page_num}] Loading image (DPI: {self.dpi})...")
        self.load_image()
        
        print(f"[Page {page_num}] Finding document boundaries...")
        contour = self.find_document_contour(self.original)
        
        print(f"[Page {page_num}] Applying perspective correction...")
        self.warped = self.perspective_transform(self.original, contour)
        
        print(f"[Page {page_num}] Preprocessing image and detecting table structure...")
        self.processed = self.preprocess_image(self.warped)
        
        # Save column structure
        self.save_column_structure(output_dir, page_num)
        
        print(f"[Page {page_num}] Performing OCR extraction (multi-pass: {multi_pass})...")
        self.results = self.perform_ocr(self.processed, page_num, multi_pass=multi_pass)
        
        # Count results by pass
        pass1_count = sum(1 for r in self.results if r.get('pass', 1) == 1)
        pass2_count = sum(1 for r in self.results if r.get('pass', 1) == 2)
        
        if multi_pass and pass2_count > 0:
            print(f"[Page {page_num}] Extracted {len(self.results)} text elements (Pass 1: {pass1_count}, Pass 2: {pass2_count} additional)")
        else:
            print(f"[Page {page_num}] Extracted {len(self.results)} text elements")
        
        # Add fallback contour flag to all results
        for result in self.results:
            result['used_fallback_contour'] = self.used_fallback_contour
        
        # Create visualization
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(self.image_path))[0]
        vis_filename = os.path.join(output_dir, f"{basename}_visualized.png")
        
        print(f"[Page {page_num}] Creating visualization...")
        visualization = self.visualize_ocr(self.processed, self.results)
        cv2.imwrite(vis_filename, visualization)
        print(f"[Page {page_num}] Saved {vis_filename}")
        
        return {
            'visualization': vis_filename,
            'results': self.results,
            'used_fallback': self.used_fallback_contour,
            'column_lines': self.column_lines,
            'num_columns': len(self.column_lines) + 1
        }


def estimate_missing_data(results: List[Dict], image_shape: Tuple[int, int], 
                         column_lines: List[int]) -> Dict:
    """
    Estimate how much data might be missing based on table structure.
    
    Args:
        results: OCR results for a page
        image_shape: (height, width) of the processed image
        column_lines: Detected column line positions
        
    Returns:
        Dictionary with missing data estimates
    """
    if not results:
        return {
            'estimated_rows': 0,
            'detected_rows': 0,
            'estimated_cells': 0,
            'detected_cells': len(results),
            'estimated_missing_percent': 100,
            'low_confidence_count': 0
        }
    
    # Count unique lines
    unique_lines = len(set(r['line_num'] for r in results))
    
    # Estimate expected rows based on image height
    # Assume average row height of ~40 pixels at 400 DPI
    avg_row_height = 40
    estimated_rows = max(unique_lines, image_shape[0] // avg_row_height)
    
    # Estimate expected columns (including label column)
    estimated_cols = len(column_lines) + 1 if column_lines else 10
    
    # Calculate expected vs detected cells
    estimated_cells = estimated_rows * estimated_cols
    detected_cells = len(results)
    
    # Count low confidence results (likely need manual verification)
    low_confidence_count = sum(1 for r in results if r['conf'] < 50)
    
    # Estimate missing percentage
    missing_percent = max(0, ((estimated_cells - detected_cells) / estimated_cells * 100))
    
    return {
        'estimated_rows': estimated_rows,
        'detected_rows': unique_lines,
        'estimated_cells': estimated_cells,
        'detected_cells': detected_cells,
        'estimated_missing_percent': round(missing_percent, 1),
        'low_confidence_count': low_confidence_count,
        'needs_verification_percent': round((low_confidence_count / detected_cells * 100), 1) if detected_cells > 0 else 0
    }


def group_column_structures(page_column_data: List[Dict], tolerance: int = 20) -> List[Dict]:
    """
    Group pages with similar column structures (same table continuing across pages).
    
    Args:
        page_column_data: List of dicts with page_num, num_columns, column_lines
        tolerance: Maximum pixel difference to consider columns aligned
        
    Returns:
        List of table groups with page ranges and standardized column structure
    """
    if not page_column_data:
        return []
    
    # Sort by page number
    page_column_data = sorted(page_column_data, key=lambda x: x['page_num'])
    
    groups = []
    current_group = {
        'start_page': page_column_data[0]['page_num'],
        'end_page': page_column_data[0]['page_num'],
        'num_columns': page_column_data[0]['num_columns'],
        'column_lines': page_column_data[0]['column_lines'],
        'pages': [page_column_data[0]['page_num']]
    }
    
    for i in range(1, len(page_column_data)):
        curr_data = page_column_data[i]
        
        # Check if this page has similar structure to current group
        same_num_cols = curr_data['num_columns'] == current_group['num_columns']
        
        # Check if column lines align (within tolerance)
        columns_align = False
        if same_num_cols and len(curr_data['column_lines']) == len(current_group['column_lines']):
            columns_align = all(
                abs(curr_data['column_lines'][j] - current_group['column_lines'][j]) <= tolerance
                for j in range(len(curr_data['column_lines']))
            )
        
        if same_num_cols and (columns_align or not current_group['column_lines']):
            # Same table structure - add to current group
            current_group['end_page'] = curr_data['page_num']
            current_group['pages'].append(curr_data['page_num'])
            
            # Average the column line positions for better accuracy
            if curr_data['column_lines']:
                for j in range(len(current_group['column_lines'])):
                    current_group['column_lines'][j] = int(
                        (current_group['column_lines'][j] + curr_data['column_lines'][j]) / 2
                    )
        else:
            # Different structure - start new group
            groups.append(current_group)
            current_group = {
                'start_page': curr_data['page_num'],
                'end_page': curr_data['page_num'],
                'num_columns': curr_data['num_columns'],
                'column_lines': curr_data['column_lines'],
                'pages': [curr_data['page_num']]
            }
    
    # Add the last group
    groups.append(current_group)
    
    return groups


def save_table_structure_report(groups: List[Dict], output_dir: str):
    """
    Save a report of detected table structures across pages.
    
    Args:
        groups: List of table groups
        output_dir: Directory to save report
    """
    report_file = os.path.join(output_dir, "table_structure_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TABLE STRUCTURE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write("This report shows how pages are grouped by table structure.\n")
        f.write("Pages with similar column layouts are grouped together.\n")
        f.write("Use this to reconstruct continuous tables in your spreadsheet.\n\n")
        
        for i, group in enumerate(groups, 1):
            f.write(f"\nTable Group {i}:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Pages: {group['start_page']} to {group['end_page']}")
            if len(group['pages']) > 1:
                f.write(f" ({len(group['pages'])} pages)\n")
            else:
                f.write(" (1 page)\n")
            f.write(f"  Page numbers: {group['pages']}\n")
            f.write(f"  Number of columns: {group['num_columns']}\n")
            
            if group['column_lines']:
                f.write(f"  Column separators at x-positions: {group['column_lines']}\n")
                f.write(f"\n  Column structure:\n")
                f.write(f"    Column 0: 0 to {group['column_lines'][0]} (Label/Header column)\n")
                for j in range(len(group['column_lines']) - 1):
                    f.write(f"    Column {j+1}: {group['column_lines'][j]} to {group['column_lines'][j+1]} (Data column)\n")
                f.write(f"    Column {len(group['column_lines'])}: {group['column_lines'][-1]} to end (Data column)\n")
            else:
                f.write(f"  No column lines detected (single column or unstructured)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SPREADSHEET RECONSTRUCTION GUIDE\n")
        f.write("=" * 70 + "\n")
        f.write("""
To reconstruct the tables in a spreadsheet:

1. Each Table Group represents a continuous table across multiple pages
2. Filter the CSV by page numbers in each group
3. Sort by line_num and column_index within each page
4. Place data from column_index=0 in the first spreadsheet column (labels)
5. Place data from column_index=1,2,3... in subsequent columns (data)
6. Each unique line_num becomes a new row in the spreadsheet
7. Continue rows across pages within the same Table Group

Example SQL-like query for Table Group 1:
  SELECT * FROM ocr_results 
  WHERE page_num BETWEEN [start_page] AND [end_page]
  ORDER BY page_num, line_num, column_index

This ensures data flows naturally from one page to the next within each table.
        """)
    
    print(f"\n✓ Table structure report saved: {report_file}")


def batch_process_images(image_dir: str, output_dir: str = "Processed", 
                         pattern: str = "*.png", dpi: int = 400,
                         multi_pass: bool = True) -> pd.DataFrame:
    """
    Process multiple images in batch and combine results into a single CSV.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save processed images
        pattern: Glob pattern to match image files
        dpi: DPI of input images (affects preprocessing)
        multi_pass: Use multi-pass OCR to catch single digits (recommended)
    
    Returns:
        DataFrame with all OCR results combined
    """
    image_pattern = os.path.join(image_dir, pattern)
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"No images found matching pattern: {image_pattern}")
        return pd.DataFrame()
    
    print("=" * 60)
    print(f"BATCH OCR PROCESSING")
    print("=" * 60)
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    print(f"Processing DPI: {dpi}")
    print(f"Multi-pass OCR: {'Enabled' if multi_pass else 'Disabled'}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    total_pass1 = 0
    total_pass2 = 0
    fallback_pages = []
    page_estimates = []
    page_column_data = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing image {idx}/{len(image_files)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            # Extract page number from filename
            basename = os.path.basename(image_path)
            match = re.search(r'-(\d+)', basename)
            page_num = int(match.group(1)) if match else idx
            
            # Create extractor with DPI setting and process
            extractor = BatchOCRExtractor(image_path, dpi=dpi)
            output = extractor.process(page_num=page_num, output_dir=output_dir, 
                                      multi_pass=multi_pass)
            
            if output and 'results' in output:
                all_results.extend(output['results'])
                
                # Track column structure for this page
                page_column_data.append({
                    'page_num': page_num,
                    'num_columns': output.get('num_columns', 0),
                    'column_lines': output.get('column_lines', [])
                })
                
                # Track pages with fallback contours
                if output.get('used_fallback', False):
                    fallback_pages.append(page_num)
                
                # Estimate missing data for this page
                page_results = output['results']
                image_shape = extractor.processed.shape[:2] if extractor.processed is not None else (0, 0)
                estimate = estimate_missing_data(page_results, image_shape, extractor.column_lines)
                estimate['page_num'] = page_num
                page_estimates.append(estimate)
                
                # Count pass statistics
                pass1_count = sum(1 for r in output['results'] if r.get('pass', 1) == 1)
                pass2_count = sum(1 for r in output['results'] if r.get('pass', 1) == 2)
                total_pass1 += pass1_count
                total_pass2 += pass2_count
                
                print(f"[Page {page_num}] Successfully processed {len(output['results'])} text elements")
                print(f"[Page {page_num}] Estimated missing: {estimate['estimated_missing_percent']}% | Low confidence: {estimate['low_confidence_count']}")
            
        except Exception as e:
            print(f"[Page {page_num}] Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Group pages by table structure
    table_groups = group_column_structures(page_column_data)
    save_table_structure_report(table_groups, output_dir)
    
    # Combine all results into a single DataFrame
    if all_results:
        combined_df = pd.DataFrame(all_results)
        
        # Define column order
        columns = ['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
                   'left', 'top', 'width', 'height', 'conf', 'text', 'column_index', 
                   'used_fallback_contour']
        
        # Add pass column if multi-pass was used
        if 'pass' in combined_df.columns:
            columns.append('pass')
        
        existing_columns = [col for col in columns if col in combined_df.columns]
        combined_df = combined_df[existing_columns]
        
        # Save combined CSV
        output_csv = os.path.join(output_dir, "combined_ocr_results.csv")
        combined_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Save estimates report
        estimates_df = pd.DataFrame(page_estimates)
        estimates_csv = os.path.join(output_dir, "data_completeness_report.csv")
        estimates_df.to_csv(estimates_csv, index=False)
        
        # Calculate overall statistics
        total_estimated_cells = sum(e['estimated_cells'] for e in page_estimates)
        total_detected_cells = len(all_results)
        total_low_conf = sum(e['low_confidence_count'] for e in page_estimates)
        overall_missing_percent = ((total_estimated_cells - total_detected_cells) / total_estimated_cells * 100) if total_estimated_cells > 0 else 0
        
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Total images processed: {len(image_files)}")
        print(f"Total text elements extracted: {len(all_results)}")
        if multi_pass and total_pass2 > 0:
            print(f"  Pass 1 (standard): {total_pass1}")
            print(f"  Pass 2 (digit recovery): {total_pass2}")
            print(f"  → {total_pass2} additional digits/numbers recovered!")
        
        print(f"\nDetected {len(table_groups)} distinct table(s) across pages:")
        for i, group in enumerate(table_groups, 1):
            page_range = f"pages {group['start_page']}-{group['end_page']}" if group['start_page'] != group['end_page'] else f"page {group['start_page']}"
            print(f"  Table {i}: {page_range} ({group['num_columns']} columns)")
        
        print("\n" + "-" * 60)
        print("OVERALL DATA COMPLETENESS ESTIMATE")
        print("-" * 60)
        print(f"Estimated total cells across all pages: {total_estimated_cells:,}")
        print(f"Detected cells: {total_detected_cells:,}")
        print(f"Estimated missing cells: ~{total_estimated_cells - total_detected_cells:,} ({overall_missing_percent:.1f}%)")
        print(f"Low confidence items (need verification): {total_low_conf:,} ({total_low_conf/total_detected_cells*100:.1f}% of detected)")
        print(f"\n⚠ MANUAL DATA ENTRY ESTIMATE:")
        print(f"  • Missing cells to enter: ~{total_estimated_cells - total_detected_cells:,}")
        print(f"  • Low-confidence items to verify: ~{total_low_conf:,}")
        print(f"  • Total items needing attention: ~{total_estimated_cells - total_detected_cells + total_low_conf:,}")
        print(f"  • Estimated time (at 5 sec/item): ~{(total_estimated_cells - total_detected_cells + total_low_conf) * 5 / 3600:.1f} hours")
        
        if fallback_pages:
            print(f"\n⚠ Pages using fallback contour (check quality): {fallback_pages}")
        
        print(f"\nOutput files:")
        print(f"  • OCR results: {output_csv}")
        print(f"  • Completeness report: {estimates_csv}")
        print(f"  • Table structure: {os.path.join(output_dir, 'table_structure_report.txt')}")
        print(f"  • Column definitions: {output_dir}/*_columns.txt")
        print("=" * 60)
        
        return combined_df
    else:
        print("\nNo results to save.")
        return pd.DataFrame()


def main():
    """Main execution function."""
    
    print("=" * 60)
    print("BATCH OCR PROCESSOR - 400 DPI OPTIMIZED")
    print("For Historical Data Tables (1950s Air Pollution Data)")
    print("=" * 60)
    
    image_dir = input("\nEnter image directory [default=./Image]: ").strip() or "./Image"
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found!")
        return
    
    output_dir = input("Enter output directory [default=Processed]: ").strip() or "Processed"
    pattern = input("Enter filename pattern [default=*.png]: ").strip() or "*.png"
    
    print(f"\nStarting batch processing at 400 DPI...")
    
    # Ask about multi-pass OCR
    print("\nUse multi-pass OCR to recover missing single digits?")
    print("  This runs a second OCR pass optimized for numbers.")
    print("  Recommended for tables with lots of numeric data.")
    multi_pass_input = input("Enable multi-pass? [Y/n]: ").strip().lower()
    multi_pass = multi_pass_input != 'n'
    
    results_df = batch_process_images(image_dir, output_dir, pattern, dpi=400, 
                                      multi_pass=multi_pass)
    
    if not results_df.empty:
        print(f"\n✓ Processing complete! Review the data_completeness_report.csv")
        print(f"  to prioritize manual data entry efforts.")
        




if __name__ == "__main__":
    main()