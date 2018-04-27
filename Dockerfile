FROM python:3.6-slim-stretch

ADD src/requirements.txt requirements.txt
ADD src/anonymise.py /usr/local/bin/anonymise.py

ENV CORENLP_VERSION=2018-02-27

RUN mkdir -p /usr/share/man/man1
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y imagemagick gcc g++ libsm6 tk tesseract-ocr openjdk-8-jre-headless curl unzip
RUN apt-get clean

RUN pip install --no-cache-dir spacy
RUN python -m spacy download en
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -LO http://nlp.stanford.edu/software/stanford-corenlp-full-${CORENLP_VERSION}.zip
RUN unzip stanford-corenlp-full-${CORENLP_VERSION}.zip && \
	  rm stanford-corenlp-full-${CORENLP_VERSION}.zip
RUN export CLASSPATH=$(find . -name '*.jar')

WORKDIR /docs-anonymiser
