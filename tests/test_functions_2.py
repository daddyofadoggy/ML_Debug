import pytest
import numpy as np
#import torch
#from torch.autograd import gradcheck

import sys
import os

# Directory of this file
dir_path = os.path.dirname(os.path.abspath(__file__))

# Directory of the project
project_dir = os.path.join(dir_path, "..")

# Add the project directory to the Python path
sys.path.append(project_dir)

from mlfin.functions_2 import square

@pytest.mark.parametrize("parameter,", [3])
def test_square_forward(parameter):
    x = float(parameter)
    relu = square

    output = relu(x)
    trusted_output = 9.0

    assert np.allclose(output, trusted_output), "Failed for 1D array"
