"""
File reader executor module.
"""
from typing import List

import pymupdf

from coxit.extractor.highlight_viz import visualize_paragraphs
from coxit.extractor.paragraph_parser import ParagraphData, ParagraphParser, SectionPageMapper


class FileReader:
    """
    File reader executor class.
    """

    def __init__(
            self,
            file_data: bytes,
            file_name: str,
    ) -> None:
        self._file_data = file_data
        self._file_name = file_name

    def extract_paragraphs(self, visualize: bool = False) -> List[ParagraphData]:
        """
        Extracts paragraphs from a PDF file with section information.
        """
        # Open the PDF document
        pdf_doc = pymupdf.open(stream=self._file_data, filetype="pdf")

        # Pipeline 1: Map sections to pages
        section_mapper = SectionPageMapper(self._file_data, self._file_name)
        page_to_section = section_mapper.map_sections_to_pages()

        # Pipeline 2: Extract paragraphs
        paragraph_parser = ParagraphParser(pdf_doc)
        paragraphs = paragraph_parser.extract_paragraphs()

        # Join results: Assign section numbers to paragraphs
        for paragraph in paragraphs:
            paragraph.section_number = page_to_section.get(paragraph.page_n)

        if visualize:
            output_dir = f"highlighted_paragraphs_{self._file_name}"
            visualize_paragraphs(
                file_data=self._file_data,
                file_name=self._file_name,
                paragraphs=paragraphs,
                output_dir=output_dir
            )
            print(f"Paragraph visualization saved to {output_dir}")

        return paragraphs
