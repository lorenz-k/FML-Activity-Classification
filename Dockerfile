FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-compile \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.11.0

COPY requirements-training.txt ./
RUN pip install --no-compile -r requirements-training.txt

COPY src ./src
COPY outputs/har ./outputs/har

CMD ["python", "-m", "src.flower_server"]