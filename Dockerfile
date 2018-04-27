FROM python:3.6-slim-jessie

ADD src/requirements.txt requirements.txt
ADD src/anonymise.py /usr/local/bin/anonymise.py

RUN apt-get update
RUN apt-get install -y imagemagick gcc g++
RUN apt-get clean

RUN pip install --no-cache-dir spacy
RUN python -m spacy download en
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /docs-anonymiser
