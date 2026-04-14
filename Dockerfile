FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install --fix-missing -y -q --no-install-recommends \
        libgomp1 ffmpeg libsm6 pdftohtml libxext6 git ninja-build g++ qpdf pandoc curl \
        ocrmypdf \
        tesseract-ocr-fra tesseract-ocr-spa tesseract-ocr-deu tesseract-ocr-ara \
        tesseract-ocr-mya tesseract-ocr-hin tesseract-ocr-tam tesseract-ocr-tha \
        tesseract-ocr-chi-sim tesseract-ocr-tur tesseract-ocr-ukr tesseract-ocr-ell \
        tesseract-ocr-rus tesseract-ocr-kor tesseract-ocr-kor-vert && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/src /app/models

RUN addgroup --system python && adduser --system --group python
RUN mkdir -p /home/python/.paddlex && chown -R python:python /app /home/python/.paddlex
USER python

ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.lock.txt requirements.lock.txt
COPY --chown=python:python src/patches /app/src/patches
RUN uv pip install --upgrade pip
RUN uv pip install -r requirements.lock.txt \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple/ \
    --index-strategy unsafe-best-match

RUN git clone https://github.com/facebookresearch/detectron2 /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    git checkout 70f454304e1a38378200459dd2dbca0f0f4a5ab4 && \
    uv pip install --no-build-isolation . && \
    rm -rf /tmp/detectron2
RUN uv pip install pycocotools==2.0.8

COPY --chown=python:python ./start.sh ./start.sh
RUN chmod +x ./start.sh

ENV PYTHONPATH="/app/src"
ENV TRANSFORMERS_VERBOSITY=error
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

ENTRYPOINT ["./start.sh"]
