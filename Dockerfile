FROM tensorflow/tensorflow:2.2.3-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
        python3-pip \
        && \
    rm -rf /var/lib/apt/lists/*

RUN pip install gensim \
        ipython \
        jupyter \
        matplotlib \
        nltk \
        numpy \
        pandas \        
        scikit-learn \
        spacy \
         && \
    pip3 install imutils

