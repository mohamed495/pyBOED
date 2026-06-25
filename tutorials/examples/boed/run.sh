#!/bin/bash

#OAR -n pyBOED
#OAR  -l /nodes=1/cpu=1/numa=2,walltime=24:00:00
#OAR --stdout pyBOED.out
#OAR --stderr pyBOED.err
#OAR --project pr-airsea-modeling

source /home/doumboum/pyBOED/.venv/bin/activate

python run_bounds_standard_nn.py