import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union

from pdf_features import PdfFeatures


def load_pdf_features(
    pdf_path: Union[str, Path],
    xml_path: Optional[Union[str, Path]] = None,
) -> Optional[PdfFeatures]:
    """Race-safe wrapper around ``PdfFeatures.from_pdf_path``.

    When no ``xml_path`` is provided, ``pdf-features`` falls back to a hardcoded
    shared path (``<tmpdir>/pdf_etree.xml``) and at the end of its flow it calls
    ``os.remove`` on that file. Under concurrent execution (FastAPI threadpool,
    or two sequential calls within one request while another request is in
    flight) threads race on the shared file: one thread removes it between
    another thread's ``pdftohtml`` write and subsequent read, and the reader
    gets ``FileNotFoundError`` — which ``from_poppler_etree`` silently swallows
    and returns ``None``. Callers then crash with
    ``AttributeError: 'NoneType' object has no attribute 'file_name'``.

    This wrapper always hands ``pdf-features`` a per-call unique xml path when
    the caller doesn't supply one, and cleans up the temporary file itself so
    ``remove_xml`` in ``pdf-features`` stays as no-op.
    """
    if xml_path is not None:
        return PdfFeatures.from_pdf_path(pdf_path, xml_path)

    unique_xml_path = Path(tempfile.gettempdir()) / f"pdf_etree_{uuid.uuid4().hex}.xml"
    try:
        return PdfFeatures.from_pdf_path(pdf_path, unique_xml_path)
    finally:
        if unique_xml_path.exists():
            unique_xml_path.unlink()
