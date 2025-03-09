FROM python:3.10

RUN apt-get update && apt-get install libgl1-mesa-glx -y && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 8080
ENTRYPOINT ["uwsgi", "--ini", "/app/wsgi.ini"]