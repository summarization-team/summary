#!/bin/sh
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate Summary
fi
python3 -m unittest discover -s tests
