import os
import shutil

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from pdf_features import PdfFeatures

from src.configuration import IMAGES_ROOT_PATH, XMLS_PATH


class PdfImages:
    def __init__(self, pdf_features: PdfFeatures, pdf_images: list[Image], dpi: int = 72):
        self.pdf_features: PdfFeatures = pdf_features
        self.pdf_images: list[Image] = pdf_images
        self.dpi: int = dpi
        # Per-request isolated workdir: UUID-based subdirectory under shared IMAGES_ROOT_PATH.
        # Prevents concurrent requests from racing on cleanup of a shared directory.
        self.images_dir: Path = Path(IMAGES_ROOT_PATH) / pdf_features.file_name
        self.save_images()

    def show_images(self, next_image_delay: int = 2):
        for image_index, image in enumerate(self.pdf_images):
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Page: {image_index + 1}", image_np)
            cv2.waitKey(next_image_delay * 1000)
            cv2.destroyAllWindows()

    def save_images(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        for image_index, image in enumerate(self.pdf_images):
            image_name = f"{self.pdf_features.file_name}_{image_index}.jpg"
            image.save(str(self.images_dir / image_name))

    def remove_images(self) -> None:
        shutil.rmtree(self.images_dir, ignore_errors=True)

    @staticmethod
    def from_pdf_path(pdf_path: str | Path, pdf_name: str = "", xml_file_name: str = "", dpi: int = 72):
        xml_path = None if not xml_file_name else Path(XMLS_PATH, xml_file_name)

        if xml_path and not xml_path.parent.exists():
            os.makedirs(xml_path.parent, exist_ok=True)

        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path, xml_path)

        if pdf_name:
            pdf_features.file_name = pdf_name
        else:
            pdf_name = Path(pdf_path).parent.name if Path(pdf_path).name == "document.pdf" else Path(pdf_path).stem
            pdf_features.file_name = pdf_name
        pdf_images = convert_from_path(pdf_path, dpi=dpi)
        return PdfImages(pdf_features, pdf_images, dpi)
