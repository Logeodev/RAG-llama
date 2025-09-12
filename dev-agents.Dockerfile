FROM python:3.10-slim

COPY ./src /app

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip --root-user-action=ignore \
    && pip install -r requirements.txt

VOLUME /app