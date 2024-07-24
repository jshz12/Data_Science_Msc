FROM python:3.12-slim

ADD Assignment1_DavidIniguez_JaumeSanchez.py .

RUN mkdir -p /home/app
WORKDIR /home/app
COPY . /home/app

RUN pip install requests pandas pymongo

CMD ["python", "./Assignment1_DavidIniguez_JaumeSanchez.py"]