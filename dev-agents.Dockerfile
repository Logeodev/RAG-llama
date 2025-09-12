FROM python:3.12-slim

COPY ./src /app

RUN pip install -r /app/requirements.txt

WORKDIR /app

VOLUME /app