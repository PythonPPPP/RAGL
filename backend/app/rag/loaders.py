from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


@dataclass
class LoadedDocument:
    doc_id: int
    filename: str
    text: str


def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        return "\n".join([p.text for p in doc.paragraphs])
    if suffix == ".csv":
        return path.read_text(encoding="utf-8", errors="ignore")
    # fallback
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    # light cleanup
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)
