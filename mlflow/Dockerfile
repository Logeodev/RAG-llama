FROM python:3.12-slim

RUN pip install mlflow

EXPOSE 5000

RUN mkdir -p /mlflow

WORKDIR /mlflow

COPY ./config_tracking.py .