from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from domain.PdfImages import PdfImages
from domain.PdfSegment import PdfSegment
from pdf_token_type_labels import TokenType
from rapidocr import RapidOCR
from rapid_table import ModelType, RapidTable, RapidTableInput
from configuration import REMOTE_OCR_ENABLED, service_logger
from adapters.infrastructure.remote_ocr.vllm_openai_client import ocr_table_html


def _try_load_font(size: int) -> ImageFont.ImageFont:
    """
    Best-effort font loader for drawing high-contrast placeholder markers.
    Must work in minimal containers too (fallback to PIL default).
    """
    # Common font names (availability depends on OS/container)
    candidates = [
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _bbox_center_inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    il, it, ir, ib = inner
    ol, ot, or_, ob = outer
    cx = (il + ir) / 2.0
    cy = (it + ib) / 2.0
    return (ol <= cx <= or_) and (ot <= cy <= ob)


def _draw_formula_placeholder(
    table_image: Image.Image,
    *,
    table_origin_px: tuple[int, int],
    formula_bbox_pts: tuple[float, float, float, float],
    dpi: int,
    marker: str,
) -> None:
    """
    Draw a white rectangle + black bordered marker inside table crop image.
    Coordinates are provided in PDF points (72 dpi).
    """
    left_px_origin, top_px_origin = table_origin_px
    fl, ft, fr, fb = formula_bbox_pts

    # Convert points -> pixels, and shift into crop coordinates
    fl_px = int(fl * dpi / 72) - left_px_origin
    ft_px = int(ft * dpi / 72) - top_px_origin
    fr_px = int(fr * dpi / 72) - left_px_origin
    fb_px = int(fb * dpi / 72) - top_px_origin

    # Clip and pad
    pad = max(2, int(min(fr_px - fl_px, fb_px - ft_px) * 0.05))
    fl_px -= pad
    ft_px -= pad
    fr_px += pad
    fb_px += pad

    w, h = table_image.size
    fl_px = max(0, min(w - 1, fl_px))
    ft_px = max(0, min(h - 1, ft_px))
    fr_px = max(0, min(w, fr_px))
    fb_px = max(0, min(h, fb_px))
    if fr_px <= fl_px + 1 or fb_px <= ft_px + 1:
        return

    draw = ImageDraw.Draw(table_image)
    # White out area
    draw.rectangle([fl_px, ft_px, fr_px, fb_px], fill="white")

    box_w = fr_px - fl_px
    box_h = fb_px - ft_px
    # Choose a font size that is likely to survive OCR
    font_size = max(16, min(64, int(box_h * 0.6)))
    font = _try_load_font(font_size)

    # Compute text bbox; center it
    try:
        text_bbox = draw.textbbox((0, 0), marker, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        # Fallback when textbbox isn't available
        text_w, text_h = draw.textlength(marker, font=font), font_size

    tx = fl_px + max(2, (box_w - int(text_w)) // 2)
    ty = ft_px + max(2, (box_h - int(text_h)) // 2)

    # Draw a high-contrast frame around the text
    frame_pad = 4
    fx0 = max(fl_px + 1, tx - frame_pad)
    fy0 = max(ft_px + 1, ty - frame_pad)
    fx1 = min(fr_px - 1, tx + int(text_w) + frame_pad)
    fy1 = min(fb_px - 1, ty + int(text_h) + frame_pad)
    draw.rectangle([fx0, fy0, fx1, fy1], outline="black", width=2, fill="white")
    draw.text((tx, ty), marker, fill="black", font=font)


def extract_table_format(pdf_images: PdfImages, predicted_segments: list[PdfSegment]):
    table_segments = [segment for segment in predicted_segments if segment.segment_type == TokenType.TABLE]
    if not table_segments:
        return

    if REMOTE_OCR_ENABLED:
        service_logger.info(f"Remote OCR enabled: parsing {len(table_segments)} table segments via vLLM")
        formula_segments = [segment for segment in predicted_segments if segment.segment_type == TokenType.FORMULA]

        for table_segment in table_segments:
            try:
                page_image: Image = pdf_images.pdf_images[table_segment.page_number - 1]
                tl, tt = table_segment.bounding_box.left, table_segment.bounding_box.top
                tr, tb = table_segment.bounding_box.right, table_segment.bounding_box.bottom

                left_px = int(tl * pdf_images.dpi / 72)
                top_px = int(tt * pdf_images.dpi / 72)
                right_px = int(tr * pdf_images.dpi / 72)
                bottom_px = int(tb * pdf_images.dpi / 72)

                table_image = page_image.crop((left_px, top_px, right_px, bottom_px)).copy()

                # Find formulas that are inside this table and draw placeholders
                table_bbox_pts = (tl, tt, tr, tb)
                in_table_formulas = [
                    f
                    for f in formula_segments
                    if f.page_number == table_segment.page_number
                    and _bbox_center_inside(
                        (f.bounding_box.left, f.bounding_box.top, f.bounding_box.right, f.bounding_box.bottom),
                        table_bbox_pts,
                    )
                ]
                in_table_formulas.sort(key=lambda s: (s.bounding_box.top, s.bounding_box.left))

                marker_to_formula: dict[str, PdfSegment] = {}
                for idx, formula_segment in enumerate(in_table_formulas, start=1):
                    marker = f"FORMULA_{idx:03d}"
                    marker_to_formula[marker] = formula_segment
                    _draw_formula_placeholder(
                        table_image,
                        table_origin_px=(left_px, top_px),
                        formula_bbox_pts=(
                            formula_segment.bounding_box.left,
                            formula_segment.bounding_box.top,
                            formula_segment.bounding_box.right,
                            formula_segment.bounding_box.bottom,
                        ),
                        dpi=pdf_images.dpi,
                        marker=marker,
                    )

                expected_markers = list(marker_to_formula.keys())

                # Table OCR with retries if markers are missing in the returned HTML
                html = ""
                last_html = ""
                for attempt in range(1, 4):
                    html = ocr_table_html(table_image)
                    if not html:
                        last_html = html
                        continue
                    if not expected_markers:
                        break
                    missing = [m for m in expected_markers if m not in html]
                    if not missing:
                        break
                    last_html = html
                    service_logger.warning(
                        f"Table OCR missing {len(missing)}/{len(expected_markers)} formula markers (attempt {attempt}/3). Retrying..."
                    )
                if not html:
                    html = last_html
                if not html:
                    continue

                # Replace markers with per-formula LaTeX (already parsed by formula OCR step)
                if expected_markers:
                    for marker, formula_segment in marker_to_formula.items():
                        latex = (formula_segment.text_content or "").strip()
                        if not latex:
                            continue

                        # Prefer exact marker. Also tolerate simple variants produced by OCR.
                        variants = [
                            marker,
                            marker.replace("_", " "),
                            marker.replace("_", ""),
                            marker.lower(),
                            marker.replace("_", " ").lower(),
                            marker.replace("_", "").lower(),
                        ]
                        for v in variants:
                            if v in html:
                                html = html.replace(v, latex)
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
