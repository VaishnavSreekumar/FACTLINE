"""
Local Document Parser (Docling & Layout-Aware Page Extraction).

Extracts page-aware structured blocks, tables, and paragraphs from PDF documents
with strict preservation of original 1-indexed page numbers and character text.
"""

from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel, Field
import fitz  # PyMuPDF

class ParsedBlock(BaseModel):
    block_type: str = Field(..., description="Type of block: paragraph, heading, table, list_item")
    text: str = Field(..., description="Raw text content of the block")
    page_number: int = Field(..., description="1-indexed page number")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [x0, y0, x1, y1]")

class ParsedPage(BaseModel):
    page_number: int = Field(..., description="1-indexed page number")
    text: str = Field(..., description="Full text of the page")
    blocks: List[ParsedBlock] = Field(default_factory=list, description="Structured blocks on page")
    tables: List[List[List[Optional[str]]]] = Field(default_factory=list, description="Extracted table grids")

class DoclingLocalParser:
    """
    Local layout-aware PDF parser providing page-level chunking and table extraction.
    """

    def __init__(self, use_docling_native: bool = False):
        self.use_docling_native = use_docling_native
        self._docling_converter = None
        if self.use_docling_native:
            try:
                from docling.document_converter import DocumentConverter
                self._docling_converter = DocumentConverter()
            except ImportError:
                self.use_docling_native = False

    def parse_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[ParsedPage]:
        """
        Parse PDF pages into structured page models with blocks and tables.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        parsed_pages: List[ParsedPage] = []
        doc = fitz.open(pdf_path)

        try:
            total_pages = len(doc)
            target_pages = page_numbers if page_numbers is not None else list(range(1, total_pages + 1))

            for page_num in target_pages:
                if page_num < 1 or page_num > total_pages:
                    continue

                page = doc[page_num - 1]
                page_text = page.get_text("text") or ""
                
                # Extract structured text blocks
                raw_blocks = page.get_text("blocks") or []
                blocks: List[ParsedBlock] = []
                for b in raw_blocks:
                    if len(b) >= 5:
                        b_text = b[4].strip()
                        if b_text:
                            b_type = "heading" if len(b_text.splitlines()) == 1 and len(b_text) < 60 else "paragraph"
                            blocks.append(ParsedBlock(
                                block_type=b_type,
                                text=b_text,
                                page_number=page_num,
                                bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                            ))

                # Extract table structures if present
                tables: List[List[List[Optional[str]]]] = []
                try:
                    tabs = page.find_tables()
                    for tab in tabs:
                        t_data = tab.extract()
                        if t_data:
                            cleaned_table = [[(str(cell) if cell is not None else "") for cell in row] for row in t_data]
                            tables.append(cleaned_table)
                except Exception:
                    pass

                parsed_pages.append(ParsedPage(
                    page_number=page_num,
                    text=page_text,
                    blocks=blocks,
                    tables=tables,
                ))

            return parsed_pages
        finally:
            doc.close()
