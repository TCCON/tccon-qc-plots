#!/bin/bash

conda_base="$(conda info --base 2>/dev/null)" || conda_base=false
if [ $conda_base == false ] ; then
        echo "Conda could not be found"
    exit 1
fi

conda_init_file=${conda_base}/etc/profile.d/conda.sh
mydir=$(cd `dirname $0` && pwd)
echo "source $conda_init_file" > $mydir/.init_conda

source .init_conda

conda_envs=($(conda env list | awk '{print $1}'))
env_exists=false
for e in ${conda_envs[*]}; do
    if [[ $e == tccon-qc ]]; then
        env_exists=true
        break
    fi  
done

echo $env_exists

if $env_exists; then
    echo "The tccon-qc python environment already exists; update from environment.yml"
    conda env update -f environment.yml
else
    echo "The tccon-qc python environment does not exist"
    echo "Creating tccon-qc python environment from environment.yml"
    conda env create -f environment.yml
fi

conda activate tccon-qc
pip install git+https://github.com/TCCON/tccon-qc-email.git
cd $mydir
python setup.py develop
ln -s $(which qc_plots) run-qc-plots
