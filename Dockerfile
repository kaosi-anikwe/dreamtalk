# Use an official Anaconda runtime as a parent image
FROM continuumio/anaconda3:latest

# Install required system libraries
RUN apt-get update && apt-get install -y libglu1 build-essential

# Create and activate the conda environment
RUN conda create -n dreamtalk python=3.7.0 && \
    echo "conda activate dreamtalk" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies
RUN conda activate dreamtalk && \
    pip install -r requirements.txt && \
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge && \
    conda update ffmpeg && \
    pip install urllib3==1.26.6 && \
    pip install transformers==4.28.1 && \
    pip install dlib

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Run your application script
CMD [ "python", "-u", "/serverless.py" ]
