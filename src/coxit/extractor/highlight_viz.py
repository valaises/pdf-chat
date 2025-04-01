import os
import shutil

from typing import List, Optional

import pymupdf

from core.globals import FILES_DIR
from coxit.extractor.paragraph_parser import ParagraphData


def highlight_paragraphs_on_page(
        pdf_doc: pymupdf.Document,
        page_num: int,
        paragraphs: List[ParagraphData],
        output_path: str,
        format: str = "pdf",
        highlight_colors: Optional[List[tuple]] = None,
) -> str:
    """
    Highlights paragraphs on a specific page and saves the result as a PDF or image.

    Args:
        pdf_doc: The PDF document
        page_num: The page number (0-based)
        paragraphs: List of paragraphs to highlight
        output_path: Directory to save the output file
        format: Output format ("pdf" or "png")
        highlight_colors: List of RGB color tuples for highlighting

    Returns:
        Path to the saved file
    """
    # Create a copy of the document to avoid modifying the original
    temp_doc = pymupdf.open()
    temp_doc.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)

    # Get the page
    page = temp_doc[0]

    # Default colors if none provided
    if not highlight_colors:
        highlight_colors = [
            (1, 0.8, 0.8),  # Light red
            (0.8, 1, 0.8),  # Light green
            (0.8, 0.8, 1),  # Light blue
            (1, 1, 0.8),  # Light yellow
            (1, 0.8, 1),  # Light magenta
            (0.8, 1, 1)  # Light cyan
        ]

    # Filter paragraphs for this page (convert from 1-based to 0-based page numbering)
    page_paragraphs = [p for p in paragraphs if p.page_n == page_num + 1]

    # Highlight each paragraph with a different color
    for i, paragraph in enumerate(page_paragraphs):
        # Get the bounding box
        rect = pymupdf.Rect(paragraph.paragraph_box)

        # Choose a color
        color = highlight_colors[i % len(highlight_colors)]

        # Add highlight
        page.draw_rect(rect, color=color, fill=color, width=0, fill_opacity=0.3)

        # Add border
        page.draw_rect(rect, color=(0, 0, 0), width=0.5)


    # Create output filename
    os.makedirs(output_path, exist_ok=True)
    base_filename = f"page_{page_num + 1}_paragraphs"

    if format.lower() == "pdf":
        output_file = os.path.join(output_path, f"{base_filename}.pdf")
        temp_doc.save(output_file)
    else:  # PNG
        output_file = os.path.join(output_path, f"{base_filename}.png")
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better quality
        pix.save(output_file)

    temp_doc.close()
    return output_file


def visualize_all_pages(
        pdf_path: str,
        paragraphs: List[ParagraphData],
        output_dir: str,
        format: str = "pdf"
) -> List[str]:
    """
    Highlights paragraphs on all pages of a PDF and saves the results.

    Args:
        pdf_path: Path to the PDF file
        paragraphs: List of paragraphs to highlight
        output_dir: Directory to save the output files
        format: Output format ("pdf" or "png")

    Returns:
        List of paths to the saved files
    """
    pdf_doc = pymupdf.open(pdf_path)
    output_files = []

    # Group paragraphs by page
    paragraphs_by_page = {}
    for p in paragraphs:
        page_num = p.page_n - 1  # Convert 1-based to 0-based
        if page_num not in paragraphs_by_page:
            paragraphs_by_page[page_num] = []
        paragraphs_by_page[page_num].append(p)

    # Process each page
    for page_num in range(pdf_doc.page_count):
        if page_num in paragraphs_by_page:
            output_file = highlight_paragraphs_on_page(
                pdf_doc=pdf_doc,
                page_num=page_num,
                paragraphs=paragraphs_by_page[page_num],
                output_path=output_dir,
                format=format
            )
            output_files.append(output_file)

    pdf_doc.close()
    return output_files


def visualize_paragraphs(
        file_data: bytes,
        file_name: str,
        paragraphs: List[ParagraphData],
        output_dir: str = "highlighted_paragraphs",
        format: str = "pdf"
) -> List[str]:
    """
    Visualizes paragraphs extracted from a PDF file.

    Args:
        file_data: The PDF file data as bytes
        file_name: The name of the PDF file
        paragraphs: List of paragraphs to highlight
        output_dir: Directory to save the output files
        format: Output format ("pdf" or "png")

    Returns:
        List of paths to the saved files
    """
    # Save the PDF data to a temporary file
    output_dir = os.path.join(str(FILES_DIR), output_dir)

    # Delete the output directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    temp_pdf_path = os.path.join(output_dir, file_name)

    with open(temp_pdf_path, "wb") as f:
        f.write(file_data)

    # Visualize paragraphs
    output_files = visualize_all_pages(
        pdf_path=temp_pdf_path,
        paragraphs=paragraphs,
        output_dir=output_dir,
        format=format
    )

    return output_files
