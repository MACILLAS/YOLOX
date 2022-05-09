FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get upgrade
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/MACILLAS/YOLOX.git

COPY run.py ./YOLOX

WORKDIR /usr/src/app/YOLOX

COPY . .

CMD ["python", "./run.py"]