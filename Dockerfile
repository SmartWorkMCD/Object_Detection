FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./models/ .

COPY ./app/ .

EXPOSE 8000

CMD ["python3","main.py"]
