#!/bin/sh
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate Summary
fi
python3  ../doc_processor/__init__.py $1 $2