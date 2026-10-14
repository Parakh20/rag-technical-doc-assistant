import fitz

from core.ingestion import DocumentLoader, get_source_metadata


def _write_pdf(path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 300), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_source_metadata_known_and_fallback():
    assert get_source_metadata("faa_ac_107-2a_small_uas.pdf") == {"jurisdiction": "us", "doc_type": "regulation"}
    assert get_source_metadata("random_paper.pdf") == {"jurisdiction": "n/a", "doc_type": "research"}


def test_load_pdf_returns_one_page_object_per_pdf_page(tmp_path):
    pdf = tmp_path / "faa_ac_107-2a_small_uas.pdf"
    _write_pdf(pdf, ["Maximum altitude is 400 feet.", "Keep visual line of sight."])

    pages = DocumentLoader().load_pdf(pdf)

    assert [p.page for p in pages] == [1, 2]
    assert "400 feet" in pages[0].text
    assert pages[0].source == pdf.name
    assert pages[0].jurisdiction == "us"
    assert pages[1].word_count == len(pages[1].text.split())


def test_load_directory_reads_all_pdfs_in_sorted_order(tmp_path):
    _write_pdf(tmp_path / "b.pdf", ["Second document."])
    _write_pdf(tmp_path / "a.pdf", ["First document."])

    pages = DocumentLoader().load_directory(tmp_path)

    assert [p.source for p in pages] == ["a.pdf", "b.pdf"]
