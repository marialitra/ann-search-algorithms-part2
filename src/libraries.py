import argparse
import os
import re
import numpy as np
import struct
import json

import kahip

import torch
import torch.nn as nn


import subprocess
import multiprocessing # To detect CPU count

# import from Python existing libraries
from typing import Dict, List, Tuple
from collections import Counter as counter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# import from user-defined files
from neural_net import CNNClassifier, MLPClassifier
from neural_net import mnist_train
from neural_net import sift_train
from parseFiles import load_idx_images, load_sift_vectors, parse_neighbor_file
from utils import build_csr_from_neighbors, save_output, _slug