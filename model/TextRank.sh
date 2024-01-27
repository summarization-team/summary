#!/bin/sh
ARG1=$1
ARG2=$2
ARG3=$3
python TextRank.py $1 $2 $3

# run the code with the following command line:
# ./TextRank.sh ../data/training 5 TextRankOutpt