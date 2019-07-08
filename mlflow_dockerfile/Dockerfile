FROM python:3.6-slim

ENV MLFLOW_SERVER_HOST 0.0.0.0

RUN mkdir /mlflow/
RUN pip install mlflow==0.8.0
RUN pip install azure-storage==0.36.0

WORKDIR /
COPY . /

EXPOSE 5000

CMD mlflow server --host 0.0.0.0
