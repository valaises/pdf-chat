import re

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set

import pymupdf

from core.logger import warn


@dataclass
class ParagraphData:
    """
    Data class that holds paragraph information extracted from a PDF file.
    """
    page_n: int
    section_number: Optional[str] = None
    paragraph_text: str = ""
    paragraph_box: Tuple[float, float, float, float] = field(default_factory=tuple)

    paragraph_n: int = 0
    width: float = 0.0
    height: float = 0.0
    paragraphs_overlapping: Set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        """
        Converts the dataclass to a dictionary for JSON serialization.
        """
        return {
            "page_n": self.page_n,
            "section_number": self.section_number,
            "paragraph_text": self.paragraph_text,
            "paragraph_box": self.paragraph_box
        }

    def join_with(self, other: 'ParagraphData') -> 'ParagraphData':
        """
        Joins this paragraph with another paragraph.

        The resulting paragraph will have:
        - The page number of this paragraph
        - The section number of this paragraph (if both have the same section)
        - The combined text of both paragraphs
        - A bounding box that encompasses both paragraphs

        Args:
            other: The paragraph to join with this one

        Returns:
            A new ParagraphData instance representing the joined paragraphs
        """
        if self.page_n != other.page_n:
            raise ValueError("Cannot join paragraphs from different pages")

        # Determine the section number for the joined paragraph
        section_number = self.section_number
        if self.section_number != other.section_number:
            # If sections differ, use the first one or None if it's None
            section_number = self.section_number if self.section_number is not None else other.section_number

        joined_text = f"{self.paragraph_text}\n{other.paragraph_text}"

        # Calculate the encompassing bounding box
        x0 = min(self.paragraph_box[0], other.paragraph_box[0])
        y0 = min(self.paragraph_box[1], other.paragraph_box[1])
        x1 = max(self.paragraph_box[2], other.paragraph_box[2])
        y1 = max(self.paragraph_box[3], other.paragraph_box[3])

        return ParagraphData(
            page_n=self.page_n,
            section_number=section_number,
            paragraph_text=joined_text,
            paragraph_box=(x0, y0, x1, y1)
        )


def join_paragraphs(paragraphs: List[ParagraphData]) -> List[ParagraphData]:
    """
    Joins paragraphs based on a specified condition.

    Args:
        paragraphs: List of paragraphs to process

    Returns:
        A new list with joined paragraphs
    """
    if not paragraphs:
        return []

    result = [paragraphs[0]]

    for current in paragraphs[1:]:
        previous = result[-1]

        previous.join_with(current)
        result.append(current)

    return result


def calculate_paragraph_dimensions_and_overlaps(paragraphs: List[ParagraphData]) -> List[ParagraphData]:
    """
    Calculates dimensions for each paragraph and identifies overlapping paragraphs.

    Args:
        paragraphs: List of paragraphs to process

    Returns:
        The same list of paragraphs with updated dimension and overlap information
    """
    # Assign paragraph numbers
    for i, para in enumerate(paragraphs):
        para.paragraph_n = i

        # Calculate width and height from bounding box
        if para.paragraph_box:
            para.width = para.paragraph_box[2] - para.paragraph_box[0]
            para.height = para.paragraph_box[3] - para.paragraph_box[1]

    # Find overlapping paragraphs
    for i, para1 in enumerate(paragraphs):
        for j, para2 in enumerate(paragraphs):
            if i == j or para1.page_n != para2.page_n:
                continue

            # Check if bounding boxes overlap
            if (para1.paragraph_box[0] < para2.paragraph_box[2] and
                    para1.paragraph_box[2] > para2.paragraph_box[0] and
                    para1.paragraph_box[1] < para2.paragraph_box[3] and
                    para1.paragraph_box[3] > para2.paragraph_box[1]):

                # Calculate overlap area
                overlap_width = min(para1.paragraph_box[2], para2.paragraph_box[2]) - max(para1.paragraph_box[0],
                                                                                          para2.paragraph_box[0])
                overlap_height = min(para1.paragraph_box[3], para2.paragraph_box[3]) - max(para1.paragraph_box[1],
                                                                                           para2.paragraph_box[1])
                overlap_area = overlap_width * overlap_height

                # Only consider significant overlaps (more than 10% of the smaller paragraph's area)
                para1_area = para1.width * para1.height
                para2_area = para2.width * para2.height
                min_area = min(para1_area, para2_area)

                if min_area > 0 and overlap_area / min_area > 0.1:
                    para1.paragraphs_overlapping.add(para2.paragraph_n)

    return paragraphs


def heur1_minimize_overlapping_boxes(paragraphs: List[ParagraphData], page_dimensions: Dict[int, Dict[str, float]]) -> \
List[ParagraphData]:
    """
    Applies heuristic to minimize overlapping boxes by reducing the size of paragraphs
    that are too large and overlap with multiple other paragraphs.

    Args:
        paragraphs: List of paragraphs to process
        page_dimensions: Dictionary mapping page numbers to their dimensions

    Returns:
        The same list of paragraphs with potentially modified bounding boxes
    """
    if not paragraphs:
        return paragraphs

    # Group paragraphs by page
    pages_dict = {}
    for para in paragraphs:
        if para.page_n not in pages_dict:
            pages_dict[para.page_n] = []
        pages_dict[para.page_n].append(para)

    # Process each page separately
    for page_num, page_paragraphs in pages_dict.items():
        # Get page height from dimensions dictionary
        # Adjust for 0-based vs 1-based page numbering
        page_idx = page_num - 1  # Convert from 1-based to 0-based
        if page_idx not in page_dimensions:
            continue

        page_height = page_dimensions[page_idx]['height']

        # Process each paragraph on this page
        for para in page_paragraphs:
            # Check if paragraph is very tall (> 60% of page height)
            # and overlaps with at least 3 other paragraphs
            if (para.height > page_height * 0.6 and
                    len(para.paragraphs_overlapping) >= 3):
                # Reduce the paragraph box to a small rectangle at the top-left corner
                x1, y1 = para.paragraph_box[0], para.paragraph_box[1]
                para.paragraph_box = (x1, y1, x1 + 10, y1 + 3)

                # Update dimensions
                para.width = 10
                para.height = 3

                # Clear overlapping paragraphs since we've modified the box
                para.paragraphs_overlapping.clear()

    # After modifying boxes, recalculate overlaps
    return calculate_paragraph_dimensions_and_overlaps(paragraphs)


def heur2_standardize_paragraph_width(paragraphs: List[ParagraphData], page_dimensions: Dict[int, Dict[str, float]]) -> \
List[ParagraphData]:
    """
    Standardizes the width of all paragraphs by setting the same x1 and x2 values.
    Takes the smallest x1 and the largest x2 from all paragraphs and applies them to every paragraph.

    Args:
        paragraphs: List of paragraphs to process
        page_dimensions: Dictionary mapping page numbers to their dimensions

    Returns:
        The same list of paragraphs with standardized widths
    """
    if not paragraphs:
        return paragraphs

    # Group paragraphs by page
    pages_dict = {}
    for para in paragraphs:
        if para.page_n not in pages_dict:
            pages_dict[para.page_n] = []
        pages_dict[para.page_n].append(para)

    # Process each page separately
    for page_num, page_paragraphs in pages_dict.items():
        # Find the smallest x1 and largest x2 values for this page
        min_x1 = min(para.paragraph_box[0] for para in page_paragraphs if para.paragraph_box)
        max_x2 = max(para.paragraph_box[2] for para in page_paragraphs if para.paragraph_box)

        # Apply the standardized width to all paragraphs on this page
        for para in page_paragraphs:
            if para.paragraph_box:
                y1, y2 = para.paragraph_box[1], para.paragraph_box[3]
                para.paragraph_box = (min_x1, y1, max_x2, y2)
                para.width = max_x2 - min_x1

    # After modifying boxes, recalculate overlaps
    return calculate_paragraph_dimensions_and_overlaps(paragraphs)


def heur3_ignore_header_footer_paragraphs(paragraphs: List[ParagraphData],
                                          page_dimensions: Dict[int, Dict[str, float]]) -> List[ParagraphData]:
    """
    Filters out paragraphs located in the top 10% or bottom 10% of each page,
    as these are likely headers, footers, page numbers, or other non-content elements.

    Args:
        paragraphs: List of paragraphs to process
        page_dimensions: Dictionary mapping page numbers to their dimensions

    Returns:
        A filtered list of paragraphs, excluding those in the header/footer regions
    """
    if not paragraphs:
        return paragraphs

    # Group paragraphs by page
    pages_dict = {}
    for para in paragraphs:
        if para.page_n not in pages_dict:
            pages_dict[para.page_n] = []
        pages_dict[para.page_n].append(para)

    filtered_paragraphs = []

    # Process each page separately
    for page_num, page_paragraphs in pages_dict.items():
        # Get page height from dimensions dictionary
        page_idx = page_num - 1  # Convert from 1-based to 0-based
        if page_idx not in page_dimensions:
            filtered_paragraphs.extend(page_paragraphs)  # Keep all if no dimensions
            continue

        page_height = page_dimensions[page_idx]['height']
        page_y0 = page_dimensions[page_idx]['y0']

        # Calculate the threshold values for top 20% and bottom 20%
        top_threshold = page_y0 + (page_height * 0.2)
        bottom_threshold = page_y0 + page_height - (page_height * 0.2)

        # Filter out paragraphs in the header/footer regions
        for para in page_paragraphs:
            # Check if paragraph is entirely in the top 10% or bottom 10%
            if para.paragraph_box:
                para_top = para.paragraph_box[1]
                para_bottom = para.paragraph_box[3]

                # Skip if paragraph is entirely in the top 10%
                if para_bottom <= top_threshold:
                    continue

                # Skip if paragraph is entirely in the bottom 10%
                if para_top >= bottom_threshold:
                    continue

                # Keep paragraphs that are not in header/footer regions
                filtered_paragraphs.append(para)

    # Recalculate paragraph numbers after filtering
    for i, para in enumerate(filtered_paragraphs):
        para.paragraph_n = i

    # Recalculate overlaps with the filtered set
    return calculate_paragraph_dimensions_and_overlaps(filtered_paragraphs)


def heur4_extend_non_overlapping_paragraphs(paragraphs: List[ParagraphData],
                                            page_dimensions: Dict[int, Dict[str, float]]) -> List[ParagraphData]:
    """
    Extends the vertical boundaries of non-overlapping paragraphs to fill gaps.
    For each paragraph that doesn't overlap with others on the same page:
    - Extends its top (y0) to the bottom of the previous paragraph + 1
    - Extends its bottom (y1) to the top of the next paragraph - 1

    Args:
        paragraphs: List of paragraphs to process
        page_dimensions: Dictionary mapping page numbers to their dimensions

    Returns:
        The same list of paragraphs with adjusted vertical boundaries
    """
    if not paragraphs:
        return paragraphs

    # Group paragraphs by page
    pages_dict = {}
    for para in paragraphs:
        if para.page_n not in pages_dict:
            pages_dict[para.page_n] = []
        pages_dict[para.page_n].append(para)

    # Process each page separately
    for page_num, page_paragraphs in pages_dict.items():
        # Get page dimensions
        page_idx = page_num - 1  # Convert from 1-based to 0-based
        if page_idx not in page_dimensions:
            continue

        page_y0 = page_dimensions[page_idx]['y0']
        page_height = page_dimensions[page_idx]['height']
        page_y1 = page_y0 + page_height

        # Sort paragraphs by y0 (top position) within this page
        sorted_page_paragraphs = sorted(page_paragraphs,
                                        key=lambda p: p.paragraph_box[1] if p.paragraph_box else float('inf'))

        # Process each paragraph on this page
        for i, para in enumerate(sorted_page_paragraphs):
            # Skip paragraphs that overlap with others on this page
            has_overlap_on_page = any(overlap_para_n in [p.paragraph_n for p in page_paragraphs]
                                      for overlap_para_n in para.paragraphs_overlapping)
            if has_overlap_on_page:
                continue

            x0, y0, x1, y1 = para.paragraph_box

            # Extend top boundary (y0) to previous paragraph
            if i > 0:
                prev_para = sorted_page_paragraphs[i - 1]
                new_y0 = prev_para.paragraph_box[3] + 1  # Bottom of previous + 1
            else:
                # No previous paragraph, use page top + 10%
                new_y0 = page_y0 + (page_height * 0.1)

            # Extend bottom boundary (y1) to next paragraph
            if i < len(sorted_page_paragraphs) - 1:
                next_para = sorted_page_paragraphs[i + 1]
                new_y1 = next_para.paragraph_box[1] - 1  # Top of next - 1
            else:
                # No next paragraph, use page bottom - 10%
                new_y1 = page_y1 - (page_height * 0.1)

            # Only apply if the new boundaries make sense
            if new_y0 < new_y1:
                para.paragraph_box = (x0, new_y0, x1, new_y1)
                para.height = new_y1 - new_y0

    # After modifying boxes, recalculate overlaps
    return calculate_paragraph_dimensions_and_overlaps(paragraphs)


def heur5_filter_short_paragraphs(paragraphs: List[ParagraphData]) -> List[ParagraphData]:
    """
    Filters out paragraphs that contain fewer than 10 words, as these are likely
    not substantial content paragraphs but rather headings, captions, or other
    non-paragraph elements.

    Args:
        paragraphs: List of paragraphs to process

    Returns:
        A filtered list of paragraphs, excluding those with fewer than 10 words
    """
    if not paragraphs:
        return paragraphs

    filtered_paragraphs = []

    for para in paragraphs:
        # Count words in the paragraph text
        word_count = len(para.paragraph_text.split())

        # Keep paragraphs with at least 10 words
        if word_count >= 10:
            filtered_paragraphs.append(para)

    # Recalculate paragraph numbers after filtering
    for i, para in enumerate(filtered_paragraphs):
        para.paragraph_n = i

    # Recalculate overlaps with the filtered set
    return calculate_paragraph_dimensions_and_overlaps(filtered_paragraphs)


class ParagraphParser:
    """
    Extracts paragraphs with their bounding boxes from PDF pages.

    """

    def __init__(self, pdf_doc: pymupdf.Document):
        self.pdf_doc = pdf_doc

    def extract_paragraphs(self) -> List[ParagraphData]:
        """
        Extracts paragraphs with their bounding boxes from all pages.
        """
        paragraphs = []
        page_dimensions = {}

        for page_num in range(self.pdf_doc.page_count):
            page = self.pdf_doc.load_page(page_num)

            page_rect = page.rect
            page_dimensions[page_num] = {
                'width': page_rect.width,
                'height': page_rect.height,
                'x0': page_rect.x0,
                'y0': page_rect.y0,
                'x1': page_rect.x1,
                'y1': page_rect.y1
            }

            page_paragraphs = self._extract_page_paragraphs(page, page_num)
            paragraphs.extend(page_paragraphs)

        # applying heuristics

        paragraphs = calculate_paragraph_dimensions_and_overlaps(paragraphs)

        paragraphs = heur1_minimize_overlapping_boxes(paragraphs, page_dimensions)
        paragraphs = heur2_standardize_paragraph_width(paragraphs, page_dimensions)
        paragraphs = heur3_ignore_header_footer_paragraphs(paragraphs, page_dimensions)
        paragraphs = heur4_extend_non_overlapping_paragraphs(paragraphs, page_dimensions)
        paragraphs = heur5_filter_short_paragraphs(paragraphs)

        return paragraphs

    def _extract_page_paragraphs(self, page: pymupdf.Page, page_num: int) -> List[ParagraphData]:
        """
        Extracts paragraphs with their bounding boxes from a single page.
        Paragraphs are defined as text segments separated by two or more newlines.
        """
        # Get the full text of the page
        full_text = page.get_text("text")

        # Split the text into paragraphs based on double newlines
        # This regex matches one or more newlines followed by whitespace and another newline
        paragraphs_text = re.split(r'\n\s*\n', full_text)
        paragraphs_text = [p.strip() for p in paragraphs_text if p.strip()]

        page_paragraphs = []

        for paragraph_text in paragraphs_text:
            # Skip very short paragraphs (likely headers, page numbers, etc.)
            if len(paragraph_text) < 3:
                continue

            # Find the bounding box for this paragraph by searching for its text
            # We'll search for the first few words to improve accuracy
            search_text = paragraph_text[:min(50, len(paragraph_text))]

            # Remove newlines from search text to improve matching
            search_text = search_text.replace('\n', ' ')

            # Search for the text on the page
            instances = page.search_for(search_text)

            if instances:
                # Use the first instance found
                rect = instances[0]

                # If the paragraph is longer than the search text, try to extend the bounding box
                if len(paragraph_text) > len(search_text):
                    # Search for the last few words
                    last_words = paragraph_text[-min(50, len(paragraph_text)):].replace('\n', ' ')
                    last_instances = page.search_for(last_words)

                    if last_instances:
                        # Extend the bounding box to include the last instance
                        last_rect = last_instances[0]
                        rect = pymupdf.Rect(
                            min(rect.x0, last_rect.x0),
                            min(rect.y0, last_rect.y0),
                            max(rect.x1, last_rect.x1),
                            max(rect.y1, last_rect.y1)
                        )

                paragraph = ParagraphData(
                    page_n=page_num + 1,  # 1-based page numbering
                    paragraph_text=paragraph_text.strip(),
                    paragraph_box=(rect.x0, rect.y0, rect.x1, rect.y1)
                )
                page_paragraphs.append(paragraph)
            else:
                # If we couldn't find the text, try a different approach
                # Split the paragraph into smaller chunks and search for each
                words = paragraph_text.split()
                if words:
                    # Try with the first few words
                    first_words = ' '.join(words[:min(5, len(words))])
                    first_instances = page.search_for(first_words)

                    if first_instances:
                        rect = first_instances[0]

                        # If there are more words, try with the last few
                        if len(words) > 5:
                            last_words = ' '.join(words[-min(5, len(words)):])
                            last_instances = page.search_for(last_words)

                            if last_instances:
                                last_rect = last_instances[0]
                                rect = pymupdf.Rect(
                                    min(rect.x0, last_rect.x0),
                                    min(rect.y0, last_rect.y0),
                                    max(rect.x1, last_rect.x1),
                                    max(rect.y1, last_rect.y1)
                                )

                        paragraph = ParagraphData(
                            page_n=page_num + 1,
                            paragraph_text=paragraph_text.strip(),
                            paragraph_box=(rect.x0, rect.y0, rect.x1, rect.y1)
                        )
                        page_paragraphs.append(paragraph)
                    else:
                        # Last resort: use the whole page
                        warn(f"Could not find position for paragraph: {paragraph_text[:50]}...")
                        paragraph = ParagraphData(
                            page_n=page_num + 1,
                            paragraph_text=paragraph_text.strip(),
                            paragraph_box=(0, 0, page.rect.width, page.rect.height)
                        )
                        page_paragraphs.append(paragraph)

        return page_paragraphs
