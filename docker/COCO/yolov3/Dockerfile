from alectio/pytorch_base:latest

COPY cocodata.py /src
COPY hyperparameters.py /src
COPY main.py /src
COPY yolov3.cfg /src
COPY flow.py /src
COPY model.py /src
COPY samplers.py /src
COPY requirements.txt /src

RUN pip install -r requirements.txt

ADD utils /src/utils


