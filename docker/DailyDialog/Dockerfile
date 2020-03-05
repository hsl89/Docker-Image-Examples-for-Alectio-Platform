from alectio/pytorch_base:latest

RUN mkdir /vector

COPY dataset.py /src
COPY envs.py /src
COPY main.py /src
COPY model.py /src
COPY flow.py /src
COPY model.py /src
COPY utils.py /src

RUN pip install spacy
RUN python -m spacy download en


