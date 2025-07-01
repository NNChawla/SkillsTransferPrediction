#!/bin/bash

apt-get --assume-yes update
apt-get --assume-yes upgrade
apt-get --assume-yes install sshpass
apt-get --assume-yes install unzip
echo "Container ID is $CONTAINER_ID"
mkdir /root/workspace
mkdir /root/workspace/experimentData
mkdir /root/workspace/results
cd /root/workspace
# pip install vastai
pip install pandas
pip install numpy
pip install scikit-learn
pip install tqdm
pip install tqdm-joblib
pip install mlxtend
pip install scipy
pip install optuna
# wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
echo "Downloading run_quick_experiment.py"
scp nayan@app.arcadea.us:/srv/STP/run_quick_experiment.py ./run_quick_experiment.py
echo "Downloading send_outputs.py"
scp nayan@app.arcadea.us:/srv/STP/send_outputs.py ./send_outputs.py
echo "Downloading experimentData"
scp -r nayan@app.arcadea.us:/srv/STP/experimentData ./experimentData
echo "Running run_quick_experiment.py"
python3 run_quick_experiment.py
echo "Running send_outputs.py"
python3 send_outputs.py