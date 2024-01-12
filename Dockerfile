FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y libglu1 build-essential

WORKDIR /app

COPY . .

RUN conda env create --file environment.yml --prefix /opt/conda/envs/dreamtalk

RUN /bin/bash -c "source activate dreamtalk"

CMD ["conda", "run", "--no-capture-output", "-n", "dreamtalk", "python", "serverless.py"]
