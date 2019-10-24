# Proyect Name
ARA (Autonomous Robot for Autism)

# General Info

ARA is intended to serve as a therapeutic solution for children in the Autism Spectrum Disorder. ARA implements an emotion recognition, face and hand tracking algorithm called Autism Learning through Vision (ALV) which is a cascade classifier + convNN classifier based, which aims to work toghether with an OAC (Output Adaptive Controller) implementing a High Order Sliding Mode (HOSM) observer to control ARA's trajectories and interactions with children.

# Technologies

Husarion CORE2 - ROS w/Raspberry Pi 3 B+ & Ubuntu 16.04

Python 2.7

# Installation Requirements

Install the following packages on the Husarion CORE2 - ROS board:

    #ROS Kinetick KEY update
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
    
    #update & upgrade packages, OpenCV will be installed with the ROS update
    sudo apt-get update && apt-get upgrade
    
    #install pip v19   
    curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
    sudo python get-pip.py
    
    #install numpy requirements
    sudo apt-get install python-dev
    sudo apt install libatlas-base-dev
    
    #virtualenv system-wide install
    sudo pip install -U virtualenv
    
    #tensorflow virtualenv creation
    vitutalenv --system-site-packages -p python tensorflow
    
    #deploy tensorflow virtualenv
    source ~/tensorflow/bin/activate
    
    #install tensorflow
    pip install --upgrade pip
    sudo pip install 'https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0-buster/tensorflow-1.14.0-cp27-none-linux_armv7l.whl'
    
    #install keras
    sudo pip install keras
    
    #deactivate tensorflow virtualenv
    deactivate
#Try
install tensorflow solving the swap problem
https://gist.github.com/EKami/9869ae6347f68c592c5b5cd181a3b205
problem solution with bazel
https://github.com/bazelbuild/bazel/commit/cc8e7166e29fee39d44e578cf98a06486084a6bd

# Setup

    cd ARA-3
    python ReconocerCara.py
 
