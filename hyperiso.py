#!/usr/bin/env python3

import os, time, scipy.io
import numpy as np
import rawpy
import glob

from hyperiso.model import Model
from hyperiso.preprocess import augment

DATASET_PATH = "./datasets/canon/"

filenames = glob.glob(DATASET_DIR + "noisy/*.CR2")

learning_rate = 1e-4

model = Model()

for 
