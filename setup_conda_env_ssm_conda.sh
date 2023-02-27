#!/bin/bash

conda create -n ssm_conda_mazzu python=3.8
source activate ssm_conda_mazzu

pip install --upgrade pip
conda install -y numpy cython scipy scikit-learn matplotlib imageio pandas seaborn ipywidgets jupyterlab multiprocessing joblib
jupyter nbextension enable --py widgetsnbextension


#pip install autograd

#pip install tensorflow==1.7.0
#conda install -c conda-forge jupyterlab

#conda install -c conda-forge gcc 
xcode-select --install #installs compiler g++ from Xcode
# git clone git@github.com:lindermanlab/ssm.git
git clone git@github.com:mazzulab/ssm.git
cd ssm
pip install -e .


conda deactivate
echo 'Done with setup'
