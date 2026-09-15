FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    NODE_PATH=/opt/node/node_modules \
    PATH=/opt/node/node_modules/.bin:$PATH

ARG DEBIAN_MIRROR=https://deb.debian.org/debian
ARG DEBIAN_SECURITY_MIRROR=https://deb.debian.org/debian-security
RUN sed -i "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g; s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm bash build-essential ripgrep jq file unzip zip sqlite3 \
        poppler-utils pandoc libreoffice-writer libreoffice-calc libreoffice-impress \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
        fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/personalops/requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir -r /opt/personalops/requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/personalops/python-packages.lock
ARG PLAYWRIGHT_DOWNLOAD_HOST
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /opt/playwright

# Build-time dependencies only; every Worker/Reviewer still uses --network none.
RUN npm install --prefix /opt/node --save-exact \
        typescript@5.9.3 prettier@3.6.2 esbuild@0.25.10 \
        react@19.1.1 react-dom@19.1.1 jsdom@26.1.0 \
    && chmod -R a+rX /opt/node
COPY read_document.py /opt/personalops/read_document.py
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

RUN command -v bash >/dev/null 2>&1 \
    || (apt-get update \
        && apt-get install -y --no-install-recommends bash \
        && rm -rf /var/lib/apt/lists/*)

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --no-create-home agent \
    && mkdir -p /workspace /review \
    && chown -R 10001:10001 /workspace /review

USER 10001:10001
WORKDIR /workspace

CMD ["sh", "-lc", "while true; do sleep 3600; done"]
