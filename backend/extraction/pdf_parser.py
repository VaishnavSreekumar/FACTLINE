"""Page-aware PDF parser implementing PyMuPDF extraction."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Union

import pymupdf

from backend.models.document import PageText, ParsedDocument


class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed due to format corruption or invalid input."""
    pass


class PDFParser:
    """Extracts text and page-level metadata from PDF documents using PyMuPDF."""

    def __init__(self):
        pass

    @staticmethod
    def _compute_hash(data: bytes) -> str:
        """Generates a deterministic document ID from file content bytes."""
        return hashlib.sha256(data).hexdigest()[:16]

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Standardizes line breaks while preserving exact words, numbers, and symbols.
        
        Preserves original casing, punctuation, and character sequences.
        """
        if not raw_text:
            return ""
        # Normalize CRLF to LF without stripping or modifying content semantics
        return raw_text.replace("\r\n", "\n").replace("\r", "\n")

    def parse(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        document_name: Optional[str] = None,
    ) -> ParsedDocument:
        """Parses a PDF document from a file path into 1-indexed PageText records.

        Args:
            file_path: Path to the PDF file on disk.
            document_id: Optional custom identifier. If omitted, computed deterministically from file content.
            document_name: Optional document label. Defaults to the base filename.

        Returns:
            ParsedDocument with 1-indexed page records.

        Raises:
            FileNotFoundError: If the target file path does not exist.
            PDFParsingError: If the file is not a valid or readable PDF, or contains 0 pages.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        try:
            with open(path, "rb") as f:
                content = f.read()
        except Exception as e:
            raise PDFParsingError(f"Failed to read file from disk: {e}") from e

        doc_name = document_name or path.name
        return self.parse_bytes(content=content, document_name=doc_name, document_id=document_id)

    def parse_bytes(
        self,
        content: bytes,
        document_name: str = "document.pdf",
        document_id: Optional[str] = None,
    ) -> ParsedDocument:
        """Parses a PDF document from raw bytes.

        Args:
            content: Raw PDF bytes.
            document_name: Label or filename for the document.
            document_id: Optional identifier. If omitted, computed as SHA-256 prefix of bytes.

        Returns:
            ParsedDocument containing 1-indexed page entries.

        Raises:
            PDFParsingError: If bytes do not form a valid PDF or the document is empty.
        """
        if not content:
            raise PDFParsingError("PDF content bytes are empty.")

        doc_id = document_id or self._compute_hash(content)

        doc = None
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as e:
            raise PDFParsingError(f"PyMuPDF failed to open PDF document: {e}") from e

        try:
            total_pages = len(doc)
            if total_pages == 0:
                raise PDFParsingError(f"PDF document '{document_name}' contains 0 pages.")

            pages = []
            for page_index in range(total_pages):
                page = doc[page_index]
                # 1-indexed page numbering invariant for FACTLINE
                page_number = page_index + 1

                extracted_text = page.get_text()
                cleaned_text = self._clean_text(extracted_text)

                has_text = bool(cleaned_text.strip())
                char_count = len(cleaned_text)

                pages.append(
                    PageText(
                        page_number=page_number,
                        text=cleaned_text,
                        char_count=char_count,
                        has_text=has_text,
                    )
                )

            return ParsedDocument(
                document_id=doc_id,
                document_name=document_name,
                total_pages=total_pages,
                pages=pages,
            )
        finally:
            if doc is not None:
                doc.close()
