from PIL import Image
from domain.PdfImages import PdfImages
from domain.PdfSegment import PdfSegment
from pdf_token_type_labels import TokenType
from rapidocr import RapidOCR
from rapid_table import ModelType, RapidTable, RapidTableInput
from configuration import REMOTE_OCR_ENABLED, service_logger
from adapters.infrastructure.remote_ocr.vllm_openai_client import ocr_table_html


def extract_table_format(pdf_images: PdfImages, predicted_segments: list[PdfSegment]):
    table_segments = [segment for segment in predicted_segments if segment.segment_type == TokenType.TABLE]
    if not table_segments:
        return

    if REMOTE_OCR_ENABLED:
        service_logger.info(f"Remote OCR enabled: parsing {len(table_segments)} table segments via vLLM")
        for table_segment in table_segments:
            try:
                page_image: Image = pdf_images.pdf_images[table_segment.page_number - 1]
                left, top = table_segment.bounding_box.left, table_segment.bounding_box.top
                right, bottom = table_segment.bounding_box.right, table_segment.bounding_box.bottom
                left = int(left * pdf_images.dpi / 72)
                top = int(top * pdf_images.dpi / 72)
                right = int(right * pdf_images.dpi / 72)
                bottom = int(bottom * pdf_images.dpi / 72)
                table_image = page_image.crop((left, top, right, bottom))
                html = ocr_table_html(table_image)
                if not html:
                    continue
                table_segment.text_content = html
            except Exception as e:
                service_logger.warning(f"Remote table OCR failed: {e}")
                continue
        return

    input_args = RapidTableInput(model_type=ModelType["SLANETPLUS"])

    ocr_engine = RapidOCR()
    table_engine = RapidTable(input_args)

    for table_segment in table_segments:
        page_image: Image = pdf_images.pdf_images[table_segment.page_number - 1]
        left, top = table_segment.bounding_box.left, table_segment.bounding_box.top
        right, bottom = table_segment.bounding_box.right, table_segment.bounding_box.bottom
        left = int(left * pdf_images.dpi / 72)
        top = int(top * pdf_images.dpi / 72)
        right = int(right * pdf_images.dpi / 72)
        bottom = int(bottom * pdf_images.dpi / 72)
        table_image = page_image.crop((left, top, right, bottom))
        ori_ocr_res = ocr_engine(table_image)
        if not ori_ocr_res.txts:
            continue
        ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
        table_result = table_engine(table_image, ocr_results=ocr_results)
        table_segment.text_content = table_result.pred_html
