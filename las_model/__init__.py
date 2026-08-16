"""Python port of Liou et al. (2020) LAS-Model focal seizure model.

Source: https://github.com/jyunyouliou/LAS-Model  (MATLAB)
Paper:  eLife 2020;9:e50927  doi:10.7554/eLife.50927
"""

from .kernels import conv_field, gaussian_kernel_1d, gaussian_kernel_2d
from .model import ConvProjection, ExternalInput, GlobalProjection, MeanFieldModel
from .params import (
    MeanFieldParams,
    exp1_template,
    exp2_template,
    gen_param,
    mean_field_1d_standard_template,
    mean_field_2d_standard_template,
)
from .run import run_exp1, run_exp2

__all__ = [
    "MeanFieldModel",
    "MeanFieldParams",
    "ConvProjection",
    "GlobalProjection",
    "ExternalInput",
    "conv_field",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
    "gen_param",
    "exp1_template",
    "exp2_template",
    "mean_field_1d_standard_template",
    "mean_field_2d_standard_template",
    "run_exp1",
    "run_exp2",
]
