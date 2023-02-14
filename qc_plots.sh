#!/bin/bash
cd `dirname $0`
source .init_conda
conda activate tccon-qc
python -m qc_plots $*
