"""
PDF to High-Resolution Image Converter for OCR
Converts PDF pages to optimal DPI images for historical document processing
"""

import os
from pdf2image import convert_from_path
from PIL import Image
import glob


def convert_pdf_to_images(pdf_path: str, 
                          output_dir: str = "Image",
                          dpi: int = 300,
                          image_format: str = "PNG",
                          prefix: str = "page") -> list:
    """
    Convert PDF pages to high-resolution images optimized for OCR.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save converted images
        dpi: Resolution for conversion (300-400 recommended for OCR)
             - 300 DPI: Good balance of quality and file size
             - 400 DPI: Better for degraded historical documents
             - 600 DPI: Maximum quality but large files
        image_format: Output format (PNG, TIFF, or JPEG)
        prefix: Prefix for output filenames
    
    Returns:
        List of paths to created image files
    """
    
    # Validate PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PDF TO IMAGE CONVERTER FOR OCR")
    print("=" * 70)
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {dpi} DPI")
    print(f"Format: {image_format}")
    print("=" * 70)
    
    # Convert PDF to images
    print("\nConverting PDF pages to images...")
    print("This may take a few minutes for large PDFs...\n")
    
    try:
        # Convert with high DPI for OCR quality
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=image_format.lower(),
            thread_count=4,  # Use multiple threads for faster conversion
            grayscale=False  # Keep color for preprocessing flexibility
        )
        
        print(f"Successfully converted {len(images)} pages")
        
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure poppler is installed:")
        print("   - macOS: brew install poppler")
        raise
    
    # Save images
    output_paths = []
    
    for i, image in enumerate(images, start=1):
        # Create filename with zero-padded page numbers
        filename = f"{prefix}-{i:03d}.{image_format.lower()}"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        image.save(filepath, image_format)
        output_paths.append(filepath)
        
        # Show progress
        print(f"[{i}/{len(images)}] Saved: {filename} ({image.size[0]}x{image.size[1]} pixels)")
    
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"Total pages converted: {len(output_paths)}")
    print(f"Images saved to: {output_dir}/")
    print("=" * 70)
    
    return output_paths


def analyze_image_quality(image_dir: str, pattern: str = "*.png"):
    """
    Analyze converted images to check if they're suitable for OCR.
    
    Args:
        image_dir: Directory containing images
        pattern: File pattern to match
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, pattern)))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print("\n" + "=" * 70)
    print("IMAGE QUALITY ANALYSIS")
    print("=" * 70)
    
    total_size = 0
    
    for img_path in image_files[:5]:  # Check first 5 images
        img = Image.open(img_path)
        file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
        total_size += file_size
        
        print(f"\n{os.path.basename(img_path)}:")
        print(f"  Dimensions: {img.size[0]} x {img.size[1]} pixels")
        print(f"  Mode: {img.mode}")
        print(f"  File size: {file_size:.2f} MB")
        
        # Estimate DPI (assuming 8.5" x 11" page)
        estimated_dpi_w = img.size[0] / 8.5
        estimated_dpi_h = img.size[1] / 11
        print(f"  Estimated DPI: {estimated_dpi_w:.0f} x {estimated_dpi_h:.0f}")
    
    if len(image_files) > 5:
        print(f"\n... and {len(image_files) - 5} more images")
    
    avg_size = (total_size / min(5, len(image_files)))
    estimated_total = avg_size * len(image_files)
    
    print("\n" + "-" * 70)
    print(f"Total images: {len(image_files)}")
    print(f"Average file size: {avg_size:.2f} MB")
    print(f"Estimated total size: {estimated_total:.2f} MB")
    print("=" * 70)
    
    # Quality recommendations
    print("\nQUALITY RECOMMENDATIONS:")
    if estimated_dpi_w < 250:
        print("⚠ WARNING: DPI appears low. Consider reconverting at 300+ DPI")
    elif estimated_dpi_w < 350:
        print("✓ Good quality for standard OCR (300 DPI range)")
    else:
        print("✓ High quality - excellent for degraded documents")
    
    print("\nFor 1950s historical documents with:")
    print("  - Faded text: Use 400 DPI")
    print("  - Small text: Use 400 DPI")
    print("  - Good condition: 300 DPI is sufficient")
    print("=" * 70)


def main():
    """Main execution function."""
    
    print("\nPDF TO IMAGE CONVERTER FOR OCR")
    print("Optimized for historical documents\n")
    
    # Get PDF path
    pdf_path = input("Enter PDF file path: ").strip()
    
    if not pdf_path:
        print("Error: No PDF path provided")
        return
    
    # Remove quotes if present (from drag-and-drop)
    pdf_path = pdf_path.strip('"').strip("'")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Get output directory
    output_dir = input("Enter output directory [default=Image]: ").strip() or "Image"
    
    # Get DPI
    print("\nRecommended DPI settings:")
    print("  300 - Standard quality (faster, smaller files)")
    print("  400 - High quality (recommended for 1950s documents)")
    print("  600 - Maximum quality (slower, larger files)")
    
    dpi_input = input("Enter DPI [default=400]: ").strip()
    dpi = int(dpi_input) if dpi_input else 400
    
    # Get format
    print("\nImage format:")
    print("  PNG - Lossless, best for OCR (recommended)")
    print("  TIFF - Lossless, archival quality")
    print("  JPEG - Compressed, smaller files (not recommended)")
    
    format_input = input("Enter format [default=PNG]: ").strip().upper() or "PNG"
    
    # Get prefix
    prefix = input("Enter filename prefix [default=page]: ").strip() or "page"
    
    # Convert PDF
    try:
        output_paths = convert_pdf_to_images(
            pdf_path=pdf_path,
            output_dir=output_dir,
            dpi=dpi,
            image_format=format_input,
            prefix=prefix
        )
        
        # Analyze quality
        analyze_image_quality(output_dir, f"{prefix}-*.{format_input.lower()}")
        
        print(f"\n✓ Ready for OCR processing!")
        print(f"  Run the batch OCR script on the '{output_dir}' directory")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()