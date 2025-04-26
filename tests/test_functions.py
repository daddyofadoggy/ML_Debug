import pytest
import torch
from torch.autograd import gradcheck

import sys
import os

# Directory of this file
dir_path = os.path.dirname(os.path.abspath(__file__))

# Directory of the project
project_dir = os.path.join(dir_path, "..")

# Add the project directory to the Python path
sys.path.append(project_dir)

from mlfin.functions import Relu, ActFunc


@pytest.fixture(autouse=True)
def default_dtype() -> None:
    """Fixture to set the default dtype for PyTorch."""
    torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def default_seed() -> None:
    """Fixture to set the seed for reproducibility."""
    torch.manual_seed(321)


@pytest.mark.parametrize("shape,", [(3, 3), (4, 2), (10, 10, 10)])
def test_relu_forward(shape):
    x = torch.randn(shape)
    relu = Relu.apply

    output = relu(x)
    trusted_output = torch.nn.ReLU()(x)

    torch.allclose(output, trusted_output, atol=1e-10)


@pytest.mark.parametrize("shape,", [(3, 3), (4, 2), (10, 10, 10)])
def test_relu_gradcheck(shape):
    n_iters = 5
    for i in range(n_iters):
        inputs = (torch.randn(shape, requires_grad=True) + i, )
        gradcheck(
            Relu.apply,
            inputs,
            eps=1e-6,
            atol=1e-4,
        )


@pytest.mark.parametrize("shape,", [(3, 3), (4, 2), (10, 10, 10)])
def test_act_forward(shape):
    x = torch.randn(shape)
    act_func = ActFunc.apply

    output = act_func(x)
    trusted_output = x * torch.special.expit(x)
    torch.allclose(output, trusted_output, atol=1e-10)


@pytest.mark.parametrize("shape,", [(3, 3), (4, 2), (10, 10, 10)])
def test_act_func_gradcheck_input(shape):
    n_iters = 5
    for i in range(n_iters):
        x = torch.randn(shape, requires_grad=True) + i

        gradcheck(
            ActFunc.apply,
            x,
            eps=1e-6,
            atol=1e-4,
        )
