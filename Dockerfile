FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ src/
COPY data/ data/
COPY config/ config/
COPY .env.example .env.example

RUN mkdir -p logs

EXPOSE 8010

ENV PYTHONUNBUFFERED=1
ENV ENV=prd

CMD ["python3", "-m", "src.main.python.main"]
