import logging
import os
from pathlib import Path


SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()

handlers = [logging.StreamHandler()]
logging.root.handlers = []
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
service_logger = logging.getLogger(__name__)

RESTART_IF_NO_GPU = os.environ.get("RESTART_IF_NO_GPU", "false").lower().strip() == "true"

# Remote OCR (tables & formulas) via OpenAI-compatible HTTP endpoint (e.g. vLLM)
REMOTE_OCR_ENABLED = os.environ.get("REMOTE_OCR_ENABLED", "false").lower().strip() == "true"
REMOTE_OCR_BASE_URL = os.environ.get("REMOTE_OCR_BASE_URL", "http://vllm-ocr:8000/v1").strip()
REMOTE_OCR_API_KEY = os.environ.get("REMOTE_OCR_API_KEY", "123").strip()
REMOTE_OCR_MODEL = os.environ.get("REMOTE_OCR_MODEL", "nanonets/Nanonets-OCR2-1.5B-exp").strip()
REMOTE_OCR_TEMPERATURE = float(os.environ.get("REMOTE_OCR_TEMPERATURE", "0.5"))
REMOTE_OCR_TIMEOUT_SEC = float(os.environ.get("REMOTE_OCR_TIMEOUT_SEC", "120"))
REMOTE_OCR_MAX_CONCURRENCY = max(1, int(os.environ.get("REMOTE_OCR_MAX_CONCURRENCY", "4")))

IMAGES_ROOT_PATH = Path(ROOT_PATH, "images")
WORD_GRIDS_PATH = Path(ROOT_PATH, "word_grids")
JSONS_ROOT_PATH = Path(ROOT_PATH, "jsons")
OCR_SOURCE = Path(ROOT_PATH, "ocr", "source")
OCR_OUTPUT = Path(ROOT_PATH, "ocr", "output")
OCR_FAILED = Path(ROOT_PATH, "ocr", "failed")
JSON_TEST_FILE_PATH = Path(JSONS_ROOT_PATH, "test.json")
MODELS_PATH = Path(ROOT_PATH, "models")
XMLS_PATH = Path(ROOT_PATH, "xmls")

DOCLAYNET_TYPE_BY_ID = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List_Item",
    5: "Page_Footer",
    6: "Page_Header",
    7: "Picture",
    8: "Section_Header",
    9: "Table",
    10: "Text",
    11: "Title",
}
