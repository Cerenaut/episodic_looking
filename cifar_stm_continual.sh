#!/bin/bash

# Pretraining
python cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 3 4 --epochs 40

# Continual - tested learning rate 0.1 (default) or 0.01 
python cifar_main_stm_training.py --experiment-type continual --fine-classes 3 4 5 --coarse-classes 3 4
python cifar_main_stm_training.py --experiment-type continual --fine-classes 4 5 3 --coarse-classes 3 4
python cifar_main_stm_training.py --experiment-type continual --fine-classes 5 3 4 --coarse-classes 3 4
