FROM nvidia/cuda:10.2-base
CMD nvidia-smi

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
COPY app/requirements_verbose.txt /app/requirements_verbose.txt
RUN pip3 install -r /app/requirements_verbose.txt

COPY app/ /app/
WORKDIR /app
ENV NUM_EPOCHS=10
ENV MODEL_TYPE='EfficientDet'
ENV DATASET_LINK='HIDDEN'
ENV TRAIN_TIME_SEC=100
CMD ["python3", "train.py"]
