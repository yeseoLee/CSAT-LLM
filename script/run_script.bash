#!/bin/bash

cd ../code/
nohup python -u main.py --config config-8bit.yaml > &
wait
nohup python -u main.py --config config-4bit.yaml > &
