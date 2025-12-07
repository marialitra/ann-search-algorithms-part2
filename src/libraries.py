# Common libraries used across multiple files
import argparse
import os
import re
import numpy as np
import struct
import json

import kahip

import torch
import torch.nn as nn
import time


import subprocess
import multiprocessing # To detect CPU count

# Import from Python existing libraries
from typing import Dict, List, Tuple, Sequence, Optional
from collections import Counter as counter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Import from user-defined files
from dataset_utils import make_sift_dataloaders
from neural_net import CNNClassifier, MLPClassifier, mnist_train, sift_train
from parseFiles import load_idx_images, load_sift_vectors, parse_neighbor_file
from utils import build_csr_from_neighbors, save_builds_output, _slug, load_data, validate_args
from runSearchExe import build_executable, run_ivfflat
from nlsh_core import neural_lsh