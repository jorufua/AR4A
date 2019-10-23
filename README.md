# Proyect Name
ARA (Autonomous Robot for Autism)

Emotion recognition, face and hand tracking algorithm used for ARA.
# General Info

This is the ALV (Autism Learning through Vision) algorithm used in ARA to track emotions, face location and hand rising of Autistic Spectrum Disorder children which are intented to use ARA as a therapeutic solution.

# Technologies

Husarion CORE2 - ROS

Ubuntu 16.04

ROS - Kinetick

Keras

Tensorflow

OpenCV

Python 2.7

# Installation Requirements

Install the following packages on the Husarion CORE2 - ROS board:

    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 #update ROS key
    sudo apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116
   
    sudo apt-get update
    sudo apt-get upgrade
   
    curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"                                                      #get pip latest version
    sudo python get-pip.py                                                                                           #install pip
    sudo apt-get install python-dev
    sudo apt install libatlas-base-dev                                                                               #required for numpy
    sudo pip install -U virtualenv                                                                                   #system-wide install
    vitutalenv --system-site-packages -p python tensorflow                                                           #create virtualenv

    source ~/tensorflow/bin/activate
    pip install --upgrade pip
    sudo pip install 'https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0-buster/tensorflow-1.14.0-cp27-none-linux_armv7l.whl'
    sudo pip install keras


# Setup

cd ARA-3

python ReconocerCara.py
 
