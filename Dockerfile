FROM tensorflow/tensorflow:1.13.2-py3

COPY . /app
WORKDIR /app

RUN apt update && \
    apt -y --no-install-recommends install libgl1 \
                                           libsm6 \
                                           libxext6 \
                                           libxrender-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt