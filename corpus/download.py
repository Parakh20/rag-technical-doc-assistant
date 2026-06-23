"""Download the public technical-document corpus into corpus/.

Sources (verified reachable as of this writing):
  - DGCA UAS Rules, 2021 (Gazette notification)
  - DGCA CAR Section 3, Series X, Part I (RPAS operations)
  - FAA AC 107-2A (Small Unmanned Aircraft Systems)
  - A handful of arXiv UAV/drone papers, used as a stand-in for ICAO
    Doc 10019 (paywalled on the ICAO Store; the official icao.int
    mirror returns 403 with no legitimate free copy available).

Run standalone: python corpus/download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

CORPUS_DIR = Path(__file__).parent
USER_AGENT = "Mozilla/5.0 (RAG-Technical-Doc-Assistant corpus downloader)"

REGULATORY_SOURCES = {
    # NOTE: digitalsky.dgca.gov.in/assets/files/UasRules.pdf used to serve the
    # UAS Rules 2021 PDF directly; that path now serves the DigitalSky SPA
    # shell instead (dead static link), so it has been dropped.
    "dgca_car_section3_series_x_part1.pdf": (
        "https://public-prd-dgca.s3.ap-south-1.amazonaws.com/"
        "InventoryList/headerblock/drones/D3X-X1.pdf"
    ),
    "faa_ac_107-2a_small_uas.pdf": (
        "https://www.faa.gov/documentlibrary/media/advisory_circular/ac_107-2a.pdf"
    ),
}

# Fallback for the ICAO Doc 10019 slot: real arXiv papers on RPAS/UAV
# regulation and airspace integration, fetched via the arXiv API.
ARXIV_FALLBACK_QUERY = "remotely piloted aircraft systems airspace integration regulation"
ARXIV_FALLBACK_COUNT = 3


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return True
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not resp.content[:4] == b"%PDF":
            print(f"  [warn] {url} did not return a PDF (content-type={content_type})")
            return False
        dest.write_bytes(resp.content)
        print(f"  [ok] {dest.name} ({len(resp.content)} bytes)")
        return True
    except requests.RequestException as exc:
        print(f"  [fail] {url} -> {exc}")
        return False


def download_regulatory_pdfs() -> list[str]:
    print("Downloading regulatory source PDFs...")
    downloaded = []
    for filename, url in REGULATORY_SOURCES.items():
        dest = CORPUS_DIR / filename
        if download_file(url, dest):
            downloaded.append(filename)
    return downloaded


def download_arxiv_fallback(count: int = ARXIV_FALLBACK_COUNT) -> list[str]:
    """Fetch arXiv papers as a substitute for the paywalled ICAO Doc 10019."""
    import arxiv

    print(f"Fetching {count} arXiv UAV/RPAS papers as ICAO Doc 10019 substitute...")
    downloaded = []
    search = arxiv.Search(
        query=ARXIV_FALLBACK_QUERY,
        max_results=count,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    for result in client.results(search):
        safe_id = result.entry_id.split("/")[-1].replace(".", "_")
        filename = f"arxiv_{safe_id}.pdf"
        dest = CORPUS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {filename} already present")
            downloaded.append(filename)
            continue
        try:
            result.download_pdf(dirpath=str(CORPUS_DIR), filename=filename)
            print(f"  [ok] {filename} - {result.title}")
            downloaded.append(filename)
        except Exception as exc:  # noqa: BLE001 - arxiv lib raises various errors
            print(f"  [fail] {result.entry_id} -> {exc}")
    return downloaded


def main() -> int:
    downloaded = download_regulatory_pdfs()
    try:
        downloaded += download_arxiv_fallback()
    except ImportError:
        print("  [warn] arxiv package not installed, skipping arXiv fallback")

    existing_local_pdfs = [
        p.name for p in CORPUS_DIR.glob("*.pdf") if p.name not in downloaded
    ]
    if existing_local_pdfs:
        print(f"Also found {len(existing_local_pdfs)} pre-existing local PDF(s): "
              f"{existing_local_pdfs}")

    total = len(downloaded) + len(existing_local_pdfs)
    print(f"\nCorpus ready: {total} PDF(s) in {CORPUS_DIR}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
