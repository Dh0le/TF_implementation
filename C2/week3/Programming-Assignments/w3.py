import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
"""
Since the  Tensorflow has already udapted to TF2 version, many TF1 grammer might nolonger legal.

And keras is more efficent with less line of code.

To implement agian in TF1 will be meaning less.
"""
