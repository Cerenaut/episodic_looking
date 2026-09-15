#!/bin/bash

# Continual - tested LR 0.01 and 0.001
python cifar_main_ltm_fine_tuning.py --experiment-type continual --fine-classes 3 4 5 --learning-rate 0.01 --coarse-classes 2 3
python cifar_main_ltm_fine_tuning.py --experiment-type continual --fine-classes 4 5 3 --learning-rate 0.01 --coarse-classes 2 3
python cifar_main_ltm_fine_tuning.py --experiment-type continual --fine-classes 5 3 4 --learning-rate 0.01 --coarse-classes 2 3
