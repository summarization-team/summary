#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Summary
conda env update --file requirements.yml --prune