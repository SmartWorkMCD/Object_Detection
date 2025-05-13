FROM python:3.12-slim

WORKDIR /app

RUN apt-get install -y git

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/ .

EXPOSE 8000

CMD ["python3","main.py"]
