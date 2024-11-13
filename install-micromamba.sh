#!/bin/bash
which micromamba >/dev/null 2>&1
if [[ $? != 0 ]]; then
  echo "micromamba could not be found"
  exit 1
fi

eval "$(micromamba shell hook --shell bash)"
envs=($(micromamba env list | awk '{print $1}'))
env_exists=false
for e in ${envs[*]}; do
  if [[ $e == tccon-qc ]]; then
    env_exists=true
    break
  fi
done

echo "tccon-qc environment exists: $env_exists"

if $env_exists; then
    echo "The tccon-qc python environment already exists; update from environment.yml"
    micromamba update --yes --file environment.yml
else
    echo "The tccon-qc python environment does not exist"
    echo "Creating tccon-qc python environment from environment.yml"
    micromamba create --yes --file environment.yml
fi

micromamba activate tccon-qc
pip install git+https://github.com/TCCON/tccon-qc-email.git
cd $(dirname $0)
python setup.py develop
ln -s $(which qc_plots) run-qc-plots
