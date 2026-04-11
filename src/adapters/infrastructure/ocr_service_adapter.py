import logging
import os
import shutil
import subprocess
from pathlib import Path
from ports.services.ocr_service import OCRService
from configuration import OCR_SOURCE, OCR_OUTPUT, OCR_FAILED
from adapters.infrastructure.ocr.languages import iso_to_tesseract, supported_languages

logger = logging.getLogger(__name__)

OCR_ENGINE = os.environ.get("OCR_ENGINE", "paddleocr")
OCR_DOWNSAMPLE_ABOVE = int(os.environ.get("OCR_DOWNSAMPLE_ABOVE", "1200"))


class OCRServiceAdapter(OCRService):
    @staticmethod
    def _build_lang_flag(language: str) -> str:
        """Convert ISO language code(s) to ocrmypdf -l flag value.

        Supports single ('ru') and multiple ('ru+en') languages.
        For PaddleOCR only the first language is used (plugin limitation).
        For Tesseract all languages are passed as 'rus+eng'.
        """
        parts = language.replace(",", "+").split("+")
        tesseract_codes = []
        for part in parts:
            part = part.strip()
            if part in iso_to_tesseract:
                tesseract_codes.append(iso_to_tesseract[part])
            else:
                tesseract_codes.append(part)
        return "+".join(tesseract_codes)

    def process_pdf_ocr(self, filename: str, namespace: str, language: str = "en") -> Path:
        source_pdf_filepath, processed_pdf_filepath, failed_pdf_filepath = self._get_paths(namespace, filename)
        os.makedirs(processed_pdf_filepath.parent, exist_ok=True)

        lang_flag = self._build_lang_flag(language)

        if OCR_ENGINE == "paddleocr":
            cmd = [
                "ocrmypdf",
                "--plugin", "ocrmypdf_paddleocr.plugin",
                "--paddle-use-gpu",
                "-j", "1",
                "-l", lang_flag,
                str(source_pdf_filepath),
                str(processed_pdf_filepath),
                "--force-ocr",
                "--tesseract-downsample-above", str(OCR_DOWNSAMPLE_ABOVE),
            ]
        else:
            cmd = [
                "ocrmypdf",
                "-l", lang_flag,
                str(source_pdf_filepath),
                str(processed_pdf_filepath),
                "--force-ocr",
                "--tesseract-downsample-above", str(OCR_DOWNSAMPLE_ABOVE),
            ]

        logger.info("Running OCR (%s): %s", OCR_ENGINE, " ".join(cmd))
        # capture_output=True so ocrmypdf's stdout/stderr land in our logger
        # instead of disappearing into the void on failure (subprocess output
        # was previously inherited from the parent and could be lost when the
        # child was OOM-killed before flushing).
        result = subprocess.run(cmd, capture_output=True, text=True)

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            logger.info("ocrmypdf stdout for %s:\n%s", source_pdf_filepath, stdout)
        if stderr:
            log = logger.error if result.returncode != 0 else logger.info
            log("ocrmypdf stderr for %s:\n%s", source_pdf_filepath, stderr)

        if result.returncode == 0:
            logger.info("OCR finished for %s -> %s", source_pdf_filepath, processed_pdf_filepath)
            return processed_pdf_filepath

        os.makedirs(failed_pdf_filepath.parent, exist_ok=True)
        try:
            shutil.move(source_pdf_filepath, failed_pdf_filepath)
        except Exception:
            logger.exception("Could not move source pdf to failed folder: %s", source_pdf_filepath)

        raise RuntimeError(
            f"ocrmypdf failed for {source_pdf_filepath} (returncode={result.returncode})"
        )

    def get_supported_languages(self) -> list[str]:
        return supported_languages()

    def _get_paths(self, namespace: str, pdf_file_name: str) -> tuple[Path, Path, Path]:
        # NOTE: pdf_file_name may legitimately contain dots (e.g. "1. DD/13.6/foo.pdf")
        # so we cannot use split(".")[:-1] to strip the extension — Path.with_suffix
        # only touches the trailing extension and leaves the rest of the name alone.
        source_pdf_filepath = Path(OCR_SOURCE, namespace, pdf_file_name)
        processed_pdf_filepath = Path(OCR_OUTPUT, namespace, pdf_file_name).with_suffix(".pdf")
        failed_pdf_filepath = Path(OCR_FAILED, namespace, pdf_file_name)
        return source_pdf_filepath, processed_pdf_filepath, failed_pdf_filepath
