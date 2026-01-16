from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    doc_id: int
    filename: str
    chunk_id: str
    text: str


def recursive_split(text: str, max_chars: int, separators: List[str]) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sep = separators[0] if separators else "\n"
    parts = text.split(sep) if sep else [text]
    if len(parts) == 1:
        # cannot split further
        return [text[:max_chars], *recursive_split(text[max_chars:], max_chars, separators)]

    chunks: List[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.extend(recursive_split(buf, max_chars, separators[1:]))
            buf = part
    if buf:
        chunks.extend(recursive_split(buf, max_chars, separators[1:]))
    return [c.strip() for c in chunks if c.strip()]


def add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out: List[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            out.append(ch)
            continue
        prev = out[-1]
        tail = prev[-overlap:] if len(prev) > overlap else prev
        out.append((tail + "\n" + ch).strip())
    return out


def make_chunks(doc_id: int, filename: str, text: str, chunk_size: int = 900, overlap: int = 120) -> List[Chunk]:
    base = recursive_split(text, max_chars=chunk_size, separators=["\n\n", "\n", ". ", " "])
    base = add_overlap(base, overlap=overlap)
    chunks: List[Chunk] = []
    for i, t in enumerate(base):
        chunks.append(
            Chunk(
                doc_id=doc_id,
                filename=filename,
                chunk_id=f"{doc_id}:{i}",
                text=t,
            )
        )
    return chunks
