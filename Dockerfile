FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y libglu1 build-essential

RUN conda create -n dreamtalk python=3.7.0 && \
    echo "conda activate dreamtalk" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

WORKDIR /

COPY . .

RUN conda activate dreamtalk && \
    pip install -r requirements.txt && \
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge && \
    conda update ffmpeg && \
    pip install urllib3==1.26.6 && \
    pip install transformers==4.28.1 && \
    pip install dlib

CMD [ "python", "-u", "/serverless.py" ]
