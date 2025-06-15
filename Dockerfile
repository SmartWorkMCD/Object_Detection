FROM python:3.12-slim

WORKDIR /app

# THIS WORKS IN RPI
#RUN apt-get update && \
#    apt-get install -y \
#        gcc \
#        g++ \
#        git \
#        libgl1 \
#        libglib2.0-0 \
#        cmake build-essential \
#    && apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
        gcc \
        g++ \
        git \
        libgl1 \
        libglib2.0-0 \
        cmake build-essential \
        v4l-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./models/ ./models/
COPY ./app/ ./app/

EXPOSE 8000

CMD ["sh", "-c", "cd app && python3 main.py"]
