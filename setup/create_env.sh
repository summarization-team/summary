#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda remove -n Summary --all
conda env create -f requirements.yml
conda activate Summary
python -m nltk.downloader all
python -m spacy download en_core_web_sm
