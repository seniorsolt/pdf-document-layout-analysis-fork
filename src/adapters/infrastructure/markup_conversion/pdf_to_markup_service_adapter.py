import tempfile
import zipfile
import io
import json
from pathlib import Path
from typing import Optional, Union
from PIL.Image import Image
from pdf2image import convert_from_path
from starlette.responses import Response

from configuration import service_logger
from domain.SegmentBox import SegmentBox
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.Label import Label
from pdf_token_type_labels.PageLabels import PageLabels
from pdf_token_type_labels.PdfLabels import PdfLabels
from pdf_token_type_labels.TokenType import TokenType

from adapters.infrastructure.markup_conversion.OutputFormat import OutputFormat
from adapters.infrastructure.markup_conversion.ExtractedImage import ExtractedImage
from adapters.infrastructure.translation.ollama_container_manager import OllamaContainerManager
from adapters.infrastructure.translation.translate_markup_document import translate_markup


class PdfToMarkupServiceAdapter:
    def __init__(self, output_format: OutputFormat):
        self.output_format = output_format

    def convert_to_format(
        self,
        pdf_content: bytes,
        segments: list[SegmentBox],
        extract_toc: bool = False,
        dpi: int = 120,
        output_file: Optional[str] = None,
        target_languages: Optional[list[str]] = None,
        translation_model: str = "gpt-oss",
    ) -> Union[str, Response]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_pdf_path = Path(temp_file.name)

        try:
            extracted_images: list[ExtractedImage] = [] if output_file else None
            user_base_name = Path(output_file).stem if output_file else None

            content_parts = self._get_styled_content_parts(
                temp_pdf_path, segments, extract_toc, dpi, extracted_images, user_base_name
            )
            content = "".join(content_parts)

            if output_file:
                translations = {}
                if target_languages and len(target_languages) > 0 and content_parts:
                    translations = self._generate_translations(
                        segments, content_parts, target_languages, translation_model, extract_toc
                    )

                return self._create_zip_response(content, extracted_images, output_file, segments, translations)

            return content
        finally:
            if temp_pdf_path.exists():
                temp_pdf_path.unlink()

    def _create_zip_response(
        self,
        content: str,
        extracted_images: list[ExtractedImage],
        output_filename: str,
        segments: list[SegmentBox],
        translations: Optional[dict[str, str]] = None,
    ) -> Response:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(output_filename, content.encode("utf-8"))

            if extracted_images:
                base_name = Path(output_filename).stem
                pictures_dir = f"{base_name}_pictures/"

                for image in extracted_images:
                    zip_file.writestr(f"{pictures_dir}{image.filename}", image.image_data)

            if translations:
                output_path = Path(output_filename)
                for language, translated_content in translations.items():
                    translated_filename = f"{output_path.stem}_{language}{output_path.suffix}"
                    zip_file.writestr(translated_filename, translated_content.encode("utf-8"))

            base_name = Path(output_filename).stem
            segmentation_filename = f"{base_name}_segmentation.json"
            segmentation_data = self._create_segmentation_json(segments)
            zip_file.writestr(segmentation_filename, segmentation_data)

        zip_buffer.seek(0)

        zip_filename = f"{Path(output_filename).stem}.zip"
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
        )

    def _create_segmentation_json(self, segments: list[SegmentBox]) -> str:
        segmentation_data = []
        for segment in segments:
            segmentation_data.append(segment.to_dict())
        return json.dumps(segmentation_data, indent=4, ensure_ascii=False)

    def _generate_translations(
        self,
        segments: list[SegmentBox],
        content_parts: list[str],
        target_languages: list[str],
        translation_model: str,
        extract_toc: bool = False,
    ) -> dict[str, str]:
        translations = {}

        ollama_manager = OllamaContainerManager()
        if not ollama_manager.ensure_service_ready(translation_model):
            return translations

        for target_language in target_languages:
            service_logger.info(f"\033[96mTranslating content to {target_language}\033[0m")
            translated_content = translate_markup(
                ollama_manager, self.output_format, segments, content_parts, translation_model, target_language, extract_toc
            )
            translations[target_language] = translated_content

        return translations

    def _create_pdf_labels_from_segments(self, vgt_segments: list[SegmentBox]) -> PdfLabels:
        page_numbers = sorted(set(segment.page_number for segment in vgt_segments))
        page_labels: list[PageLabels] = []
        for page_number in page_numbers:
            segments_in_page = [s for s in vgt_segments if s.page_number == page_number]
            labels: list[Label] = []
            for segment in segments_in_page:
                rect = Rectangle.from_width_height(segment.left, segment.top, segment.width, segment.height)
                label = Label.from_rectangle(rect, TokenType.from_text(segment.type).get_index())
                labels.append(label)
            page_labels.append(PageLabels(number=page_number, labels=labels))
        return PdfLabels(pages=page_labels)

    def _process_picture_segment(
        self,
        segment: SegmentBox,
        pdf_images: list[Image],
        pdf_path: Path,
        picture_id: int,
        dpi: int = 72,
        extracted_images: Optional[list[ExtractedImage]] = None,
        user_base_name: Optional[str] = None,
    ) -> str:

        if extracted_images is None:
            return ""

        segment_box = Rectangle.from_width_height(segment.left, segment.top, segment.width, segment.height)
        image = pdf_images[segment.page_number - 1]
        left, top, right, bottom = segment_box.left, segment_box.top, segment_box.right, segment_box.bottom
        if dpi != 72:
            left = left * dpi / 72
            top = top * dpi / 72
            right = right * dpi / 72
            bottom = bottom * dpi / 72
        cropped = image.crop((left, top, right, bottom))

        base_name = user_base_name if user_base_name else pdf_path.stem
        image_name = f"{base_name}_{segment.page_number}_{picture_id}.png"

        img_buffer = io.BytesIO()
        cropped.save(img_buffer, format="PNG")
        extracted_images.append(ExtractedImage(image_data=img_buffer.getvalue(), filename=image_name))
        return f"<img src='{base_name}_pictures/{image_name}' alt=''>\n\n"

    def _process_table_segment(self, segment: SegmentBox) -> str:
        return segment.text + "\n\n"

    def _get_token_content(self, token: PdfToken) -> str:
        if self.output_format == OutputFormat.HTML:
            return token.content_html
        else:
            return token.content_markdown

    def _get_styled_content(self, token: PdfToken, content: str) -> str:
        if self.output_format == OutputFormat.HTML:
            styled = token.token_style.get_styled_content_html(content)
            styled = token.token_style.script_type.get_styled_content(styled)
            styled = token.token_style.list_level.get_styled_content_html(styled)
            return token.token_style.hyperlink_style.get_styled_content_html(styled)
        else:
            styled = token.token_style.get_styled_content_markdown(content)
            styled = token.token_style.script_type.get_styled_content(styled)
            styled = token.token_style.list_level.get_styled_content_markdown(styled)
            return token.token_style.hyperlink_style.get_styled_content_markdown(styled)

    def _process_title_segment(self, tokens: list[PdfToken], segment: SegmentBox) -> str:
        if not tokens:
            return ""

        title_type = tokens[0].token_style.title_type
        content = " ".join([self._get_styled_content(token, token.content) for token in tokens])
        if self.output_format == OutputFormat.HTML:
            content = title_type.get_styled_content_html(content)
        else:
            content = title_type.get_styled_content_markdown(content)
        return content + "\n\n"

    def _process_regular_segment(self, tokens: list[PdfToken], segment: SegmentBox) -> str:
        if not tokens:
            return ""
        content = " ".join(self._get_token_content(t) for t in tokens)
        return content + "\n\n"

    def _get_table_of_contents(self, vgt_segments: list[SegmentBox]) -> str:
        title_segments = [s for s in vgt_segments if s.type in {TokenType.TITLE, TokenType.SECTION_HEADER}]
        table_of_contents = "# Table of Contents\n\n"
        for segment in title_segments:
            if not segment.text.strip():
                continue
            first_word = segment.text.split()[0]
            indentation = max(0, first_word.count(".") - 1)
            content = "  " * indentation + "- " + segment.text + "\n"
            table_of_contents += content
        table_of_contents += "\n"
        return table_of_contents + "\n\n"

    def _get_styled_content_parts(
        self,
        pdf_path: Path,
        vgt_segments: list[SegmentBox],
        extract_toc: bool = False,
        dpi: int = 120,
        extracted_images: Optional[list[ExtractedImage]] = None,
        user_base_name: Optional[str] = None,
    ) -> str:
        pdf_labels: PdfLabels = self._create_pdf_labels_from_segments(vgt_segments)
        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path)
        pdf_features.set_token_types(pdf_labels)
        pdf_features.set_token_styles()

        content_parts: list[str] = []
        if extract_toc:
            content_parts.append(self._get_table_of_contents(vgt_segments))

        picture_segments = [s for s in vgt_segments if s.type == TokenType.PICTURE]
        pdf_images: list[Image] = convert_from_path(pdf_path, dpi=dpi) if picture_segments else []

        for page in pdf_features.pages:
            segments_in_page = [s for s in vgt_segments if s.page_number == page.page_number]
            picture_id = 0
            for segment in segments_in_page:
                seg_box = Rectangle.from_width_height(segment.left, segment.top, segment.width, segment.height)
                tokens_in_seg = [t for t in page.tokens if t.bounding_box.get_intersection_percentage(seg_box) > 50]

                if segment.type == TokenType.PICTURE:
                    content_parts.append(
                        self._process_picture_segment(
                            segment, pdf_images, pdf_path, picture_id, dpi, extracted_images, user_base_name
                        )
                    )
                    picture_id += 1
                elif segment.type == TokenType.TABLE:
                    content_parts.append(self._process_table_segment(segment))
                elif segment.type in {TokenType.TITLE, TokenType.SECTION_HEADER}:
                    content_parts.append(self._process_title_segment(tokens_in_seg, segment))
                elif segment.type == TokenType.FORMULA:
                    content_parts.append(segment.text + "\n\n")
                else:
                    content_parts.append(
                        self._process_regular_segment(tokens_in_seg, segment)
                    )

        return content_parts
