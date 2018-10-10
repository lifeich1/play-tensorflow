#!/bin/bash

if [ ! -d ./env ]; then
    virtualenv -p python2.7 --system-site-packages ./env
fi

source ./env/bin/activate
pip install --upgrade \
    https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
