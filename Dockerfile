# syntax=docker/dockerfile:1.7

FROM rust:1.99.1-bookworm@sha256:48b7f435b8fc6378ce43b44f15e3a47fd8919b2825f835c39bd4bd3da50af5b8 AS hfrs-builder

ENV CARGO_HOME=/usr/local/cargo

RUN cargo install --locked \
    --root /out \
    --git https://github.com/huggingface/hf-hub.git \
    --rev 473de9430b0fbc544fb1d91102180eb713f8dc87 \
    hfrs

FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS runtime

ENV HOME=/home/downloader \
    HF_DOWNLOADER_OUTPUT_DIR=/models \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    HF_TOKEN_PATH=/cache/huggingface/token \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN mkdir -p /app/script /app/configs /models /cache /home/downloader \
    && chown -R 10001:10001 /app /models /cache /home/downloader

COPY --from=hfrs-builder --chmod=0555 /out/bin/hfrs /usr/local/bin/hfrs
COPY --chown=10001:10001 script/download.py /app/script/download.py
COPY --chown=10001:10001 configs/download.txt /app/configs/download.txt

WORKDIR /app
USER 10001:10001

ENTRYPOINT ["python", "-B", "/app/script/download.py"]
