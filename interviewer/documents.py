"""Document processing and text extraction for Resumes and Job Descriptions.

Supports PDF (.pdf) and Word (.docx) formats with comprehensive validation
for file type, size limits, empty files, and unreadable/scanned content.
"""
from __future__ import annotations

import io
import os
import re
from typing import Dict, Tuple

MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


class DocumentError(ValueError):
    """Raised when a document cannot be validated or parsed."""
    pass


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable control characters."""
    # Replace non-breaking spaces and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip non-printable chars (keep standard whitespace)
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return text.strip()


def extract_text_from_pdf(content: bytes, filename: str = "document.pdf") -> str:
    """Extract readable text from PDF bytes using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentError(
            "PDF processing library (pypdf) is not installed on the server. "
            "Please run: pip install pypdf"
        )

    try:
        stream = io.BytesIO(content)
        reader = PdfReader(stream)
    except Exception as exc:
        raise DocumentError(
            f"Unable to read PDF file '{filename}'. The file may be corrupted, "
            f"damaged, or not a valid PDF: {exc}"
        ) from exc

    if reader.is_encrypted:
        try:
            # Try decrypting with empty password if permitted
            decrypted = reader.decrypt("")
            if not decrypted:
                raise DocumentError(
                    f"PDF file '{filename}' is password-protected or encrypted. "
                    f"Please provide an unencrypted PDF."
                )
        except Exception as exc:
            raise DocumentError(
                f"PDF file '{filename}' is password-protected or encrypted: {exc}"
            ) from exc

    pages_text = []
    for idx, page in enumerate(reader.pages):
        try:
            pt = page.extract_text() or ""
            if pt.strip():
                pages_text.append(pt.strip())
        except Exception:
            # Continue extracting remaining pages if a single page fails
            continue

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise DocumentError(
            f"No readable text could be extracted from '{filename}'. "
            f"If this is a scanned document or image-only PDF, please upload a "
            f"text-based PDF or DOCX file, or paste the text directly."
        )

    return _clean_text(full_text)


def extract_text_from_docx(content: bytes, filename: str = "document.docx") -> str:
    """Extract readable text from DOCX bytes using python-docx."""
    try:
        import docx
    except ImportError:
        raise DocumentError(
            "Word document processing library (python-docx) is not installed on the server. "
            "Please run: pip install python-docx"
        )

    try:
        stream = io.BytesIO(content)
        doc = docx.Document(stream)
    except Exception as exc:
        raise DocumentError(
            f"Unable to read Word document '{filename}'. The file may be corrupted, "
            f"or in an unsupported legacy format (e.g., .doc instead of .docx): {exc}"
        ) from exc

    parts = []
    # Extract regular paragraphs
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)

    # Extract tables (common in resumes)
    for table in doc.tables:
        for row in table.rows:
            row_items = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_items:
                # Remove duplicate cell contents in merged cells
                deduped = []
                for item in row_items:
                    if not deduped or item != deduped[-1]:
                        deduped.append(item)
                parts.append(" | ".join(deduped))

    full_text = "\n\n".join(parts).strip()
    if not full_text:
        raise DocumentError(
            f"No readable text could be found in Word document '{filename}'. "
            f"Please verify that the document contains text content."
        )

    return _clean_text(full_text)


def extract_document_text(
    filename: str,
    content: bytes,
    max_size: int = MAX_DOCUMENT_SIZE,
) -> Tuple[str, Dict[str, int]]:
    """Validate and extract text from an uploaded document.

    Returns:
        (extracted_text, metadata_dict)

    Raises:
        DocumentError if file type, size, or extraction fails.
    """
    if not filename or not filename.strip():
        raise DocumentError("Document filename is required.")

    filename = os.path.basename(filename.strip())
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentError(
            f"Unsupported file format '{ext}'. Allowed document formats: {allowed_str}."
        )

    if not content or len(content) == 0:
        raise DocumentError(f"Uploaded file '{filename}' is empty (0 bytes).")

    if len(content) > max_size:
        size_mb = round(len(content) / (1024 * 1024), 2)
        max_mb = round(max_size / (1024 * 1024), 2)
        raise DocumentError(
            f"File '{filename}' ({size_mb} MB) exceeds maximum allowed size of {max_mb} MB."
        )

    if ext == ".pdf":
        text = extract_text_from_pdf(content, filename=filename)
    elif ext == ".docx":
        text = extract_text_from_docx(content, filename=filename)
    else:
        raise DocumentError(f"Unsupported document format: {ext}")

    meta = {
        "filename": filename,
        "format": ext[1:].upper(),
        "size_bytes": len(content),
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }
    return text, meta


def validate_text_document(
    text: str,
    doc_label: str = "Job description",
    min_length: int = 20,
) -> str:
    """Validate and clean plain text input (e.g. pasted job description)."""
    if not text or not text.strip():
        raise DocumentError(f"{doc_label} cannot be empty.")

    cleaned = _clean_text(text)
    if len(cleaned) < min_length:
        raise DocumentError(
            f"{doc_label} is too brief ({len(cleaned)} characters). "
            f"Please provide at least {min_length} characters of descriptive content."
        )

    return cleaned
