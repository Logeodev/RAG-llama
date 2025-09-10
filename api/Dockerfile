FROM python:3.12-slim
WORKDIR /app
COPY ./api /app
RUN pip install -r ./requirements.txt
VOLUME /app
EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]