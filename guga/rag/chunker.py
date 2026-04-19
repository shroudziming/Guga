from __future__ import annotations


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []

    if chunk_size <= 0:
        return [clean]

    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    index = 0
    while index < len(clean):
        piece = clean[index : index + chunk_size].strip()
        if piece:
            chunks.append(piece)
        index += step

    return chunks
