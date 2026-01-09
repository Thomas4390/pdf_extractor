"""
PDF processing utilities.

Provides functions for converting PDFs to images and computing file hashes.
"""

import hashlib
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF


def pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 200,
    pages: list[int] | None = None,
) -> list[bytes]:
    """
    Convert a PDF file to a list of PNG images.

    Each page is rendered as a separate image at the specified DPI.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default: 200)
        pages: List of page indices to render (0-indexed). If None, renders all pages.

    Returns:
        List of PNG images as bytes

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    images: list[bytes] = []

    try:
        # Calculate zoom factor to achieve target DPI
        # PDF default is 72 DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue
            pixmap = page.get_pixmap(matrix=matrix)
            images.append(pixmap.tobytes("png"))
    finally:
        doc.close()

    return images


def get_pdf_hash(pdf_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a PDF file.

    Used for cache key generation to uniquely identify PDFs.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        SHA-256 hash as hexadecimal string

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """
    Get the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages in the PDF
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def pdf_to_text(
    pdf_path: Union[str, Path],
    pages: list[int] | None = None,
    preserve_layout: bool = True,
) -> str:
    """
    Extract text content from a PDF file.

    Uses PyMuPDF to extract text from specified pages or all pages.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page indices to extract (0-indexed). If None, extracts all pages.
        preserve_layout: If True, preserves table/column layout better (uses "blocks" extraction)

    Returns:
        Extracted text as a single string with page separators

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    text_parts: list[str] = []

    try:
        for i, page in enumerate(doc):
            if pages is not None and i not in pages:
                continue

            if preserve_layout:
                # Extract text with better table/layout preservation
                text = page.get_text("text", sort=True)
            else:
                # Simple text extraction
                text = page.get_text()

            if text.strip():
                text_parts.append(f"--- Page {i + 1} ---\n{text.strip()}")
    finally:
        doc.close()

    return "\n\n".join(text_parts)
