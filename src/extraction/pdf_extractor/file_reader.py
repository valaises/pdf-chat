"""
File reader executor module.
"""
from typing import List

import pymupdf

from extraction.pdf_extractor.highlight_viz import visualize_paragraphs
from extraction.pdf_extractor.paragraph_parser import ParagraphData, ParagraphParser


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

        paragraph_parser = ParagraphParser(pdf_doc)
        paragraphs = paragraph_parser.extract_paragraphs()

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
