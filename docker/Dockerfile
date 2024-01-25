FROM ubuntu:22.04

RUN apt update && apt upgrade

RUN apt install vim tree python3-pip curl -y

RUN mkdir "workdir"
WORKDIR "workdir"
COPY requirements_prod.txt ./requirements_prod.txt

RUN pip install -r requirements.txt

