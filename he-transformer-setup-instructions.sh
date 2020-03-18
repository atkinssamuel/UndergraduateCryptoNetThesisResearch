#!/bin/bash
# This file runs the appropriate commands to replicate the he-transformer repository located at:
# https://github.com/IntelAI/he-transformer
# You must change the permissions of the file to run. Run: chmod u+x scriptname
# Make sure you run with sudo permissions
cd ~
sudo apt install g++
sudo apt update && sudo apt install -y python3-pip virtualenv python3-numpy python3-dev python3-wheel git unzip wget sudo bash-completion build-essential cmake software-properties-common git wget patch diffutils libtinfo-dev autoconf libtool doxygen graphviz
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
sudo apt-get update && sudo apt install -y clang-9 clang-tidy-9 clang-format-9
git clone https://github.com/IntelAI/he-transformer.git
sudo apt-get clean autoclean
sudo apt-get autoremove -y
sudo pip3 install --upgrade pip setuptools virtualenv==16.1.0
sudo -H pip3 install cmake --upgrade
wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
chmod +x ./bazel-0.25.2-installer-linux-x86_64.sh 
sudo bash ./bazel-0.25.2-installer-linux-x86_64.sh 
export PATH=$PATH:~/bin
source ~/.bashrc
export HE_TRANSFORMER=$(pwd)
cd ~/he-transformer/
mkdir build
cd build
sudo cmake .. -DCMAKE_CXX_COMPILER=clang++-9 -DCMAKE_C_COMPILER=clang-9
sudo make install
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate