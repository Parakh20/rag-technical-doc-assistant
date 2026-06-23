"""PDF ingestion: load, clean, and extract per-page text + metadata.

Uses PyMuPDF (fitz) to extract text with layout information so header/footer
bands can be stripped by position and section headings can be detected by
relative font size / bold weight, rather than just taking raw page.get_text().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

HEADER_ZONE_FRAC = 0.08
FOOTER_ZONE_FRAC = 0.08
HEADING_SIZE_MULTIPLIER = 1.15
MAX_HEADING_CHARS = 120


@dataclass
class Page:
    text: str
    source: str
    page: int
    section: str
    word_count: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "word_count": self.word_count,
        }


class DocumentLoader:
    """Loads PDFs into cleaned per-page text with header/footer stripped
    and the nearest detected section heading attached."""

    def load_pdf(self, path: str | Path) -> list[Page]:
        path = Path(path)
        doc = fitz.open(path)
        body_font_size = self._estimate_body_font_size(doc)
        pages: list[Page] = []
        current_section = ""
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_height = page.rect.height
            lines, current_section = self._extract_page_lines(
                page, page_height, body_font_size, current_section
            )
            cleaned = self._clean_text("\n".join(lines))
            pages.append(
                Page(
                    text=cleaned,
                    source=path.name,
                    page=page_index + 1,
                    section=current_section,
                    word_count=len(cleaned.split()),
                )
            )
        doc.close()
        return pages

    def load_directory(self, path: str | Path) -> list[Page]:
        path = Path(path)
        all_pages: list[Page] = []
        for pdf_path in sorted(path.glob("*.pdf")):
            all_pages.extend(self.load_pdf(pdf_path))
        return all_pages

    def _extract_page_lines(
        self, page, page_height: float, body_font_size: float, current_section: str
    ) -> tuple[list[str], str]:
        kept_lines: list[str] = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y1 < page_height * HEADER_ZONE_FRAC:
                    continue
                if y0 > page_height * (1 - FOOTER_ZONE_FRAC):
                    continue
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue
                if self._is_heading(line, body_font_size, line_text):
                    current_section = line_text
                kept_lines.append(line_text)
        return kept_lines, current_section

    @staticmethod
    def _is_heading(line: dict, body_font_size: float, line_text: str) -> bool:
        if len(line_text) >= MAX_HEADING_CHARS:
            return False
        max_size = max((span["size"] for span in line["spans"]), default=0)
        is_bold = any("bold" in span["font"].lower() for span in line["spans"])
        return max_size >= body_font_size * HEADING_SIZE_MULTIPLIER or is_bold

    @staticmethod
    def _estimate_body_font_size(doc) -> float:
        sizes = [
            span["size"]
            for page in doc
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]
        if not sizes:
            return 10.0
        sizes.sort()
        return sizes[len(sizes) // 2]

    @staticmethod
    def _clean_text(text: str) -> str:
        # Fix hyphenation broken across a line wrap: "exam-\nple" -> "example"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Collapse remaining newlines/whitespace into single spaces
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()


if __name__ == "__main__":
    corpus_dir = Path(__file__).parent.parent / "corpus"
    loader = DocumentLoader()
    pages = loader.load_directory(corpus_dir)
    print(f"Loaded {len(pages)} pages from {corpus_dir}")
    if pages:
        sample = pages[len(pages) // 2]
        print(f"Sample: source={sample.source} page={sample.page} "
              f"section={sample.section!r} word_count={sample.word_count}")
        print(sample.text[:300])
