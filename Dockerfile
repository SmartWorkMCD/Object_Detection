FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        git \
        libgl1 \
        libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models/ ./models/
COPY ./app/ ./app/

EXPOSE 8000

CMD ["sh", "-c", "cd app && python3 main.py"]
