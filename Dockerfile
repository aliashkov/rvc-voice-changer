FROM python:3.10

RUN pip install --upgrade pip

WORKDIR /rvc-docker

COPY . /rvc-docker

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ADD . /rvc-docker

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 7860

COPY . .

CMD ["python3", "app.py"]