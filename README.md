# image_recognition
This is a collection of various models which attempts to train a neural network for image recognition.

### Prerequisites
A folder of images downloaded into categories.

Docker or pyenv

## Install
Install docker from docker website 

Install pyenv following instructions from https://github.com/pyenv/pyenv-virtualenv

## How to run

### With docker
```
make start
make run file=main
make stop
```

### With pyenv
Initial run
```
pyenv virtualenv 3.6.6 <name of environment>
pip3 install tensorflow h5py ipykernel jupyter matplotlib numpy pandas scipy sklearn
pip install Pillow
```
Next time running in same environment 
```
pyenv activate <name of environment>
```
