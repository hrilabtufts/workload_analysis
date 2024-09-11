FROM python:3.11.10-alpine3.19

WORKDIR /app

COPY ./requirements_docker.txt ./requirements.txt

RUN pip install -r requirements.txt


COPY ./PCPS.py ./
COPY ./server.py ./

CMD ["python", "server.py", "--port", "9995"]
EXPOSE 9995
