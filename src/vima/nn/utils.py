from collections.abc import Callable
from functools import partial
from typing import Literal

from loguru import logger
from torch import nn
from torch.nn import Embedding as _Embedding


def get_activation(activation: str | Callable | None) -> Callable:
    if not activation:
        return nn.Identity
    if callable(activation):
        return activation

    ACT_LAYER = {
        "tanh": nn.Tanh,
        "relu": partial(nn.ReLU, inplace=True),
        "leaky_relu": partial(nn.LeakyReLU, inplace=True),
        "swish": partial(nn.SiLU, inplace=True),  # SiLU is alias for Swish
        "sigmoid": nn.Sigmoid,
        "elu": partial(nn.ELU, inplace=True),
        "gelu": nn.GELU,
    }
    activation = activation.lower()
    assert activation in ACT_LAYER, f"Supported activations: {ACT_LAYER.keys()}"
    return ACT_LAYER[activation]


def get_initializer(method: str | Callable, activation: str) -> Callable:
    if isinstance(method, str):
        if not method.endswith("_"):
            method = f"{method}_"
        assert (
            getattr(nn.init, method, None) is not None
        ), f"Unknown initializer: torch.nn.init.{method}"

        if method == "orthogonal_":
            try:
                gain = nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0
            return partial(nn.init.orthogonal_, gain=gain)
        return getattr(nn.init, method)

    assert callable(method)
    return method


def get_norm_layer(norm_type: Literal["batchnorm", "layernorm"] | None) -> Callable:
    if not norm_type:
        return nn.Identity

    norm_type = norm_type.lower()
    if norm_type == "batchnorm":
        return nn.BatchNorm1d
    if norm_type == "layernorm":
        return nn.LayerNorm

    raise ValueError(f"Unsupported norm layer: {norm_type}")


class Embedding(_Embedding):
    @property
    def output_dim(self):
        return self.embedding_dim


def build_mlp(
    input_dim,
    *,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int = None,
    num_layers: int | None = None,
    activation: str | Callable = "relu",
    weight_init: str | Callable = "orthogonal",
    bias_init="zeros",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    add_input_activation: bool | str | Callable = False,
    add_input_norm: bool = False,
    add_output_activation: bool | str | Callable = False,
    add_output_norm: bool = False,
) -> nn.Sequential:
    """In other popular RL implementations, tanh is typically used with orthogonal initialization,
    which may perform better than ReLU.

    Args:
        norm_type: None, "batchnorm", "layernorm", applied to intermediate layers
        add_input_activation: whether to add a nonlinearity to the input _before_
            the MLP computation. This is useful for processing a feature from a preceding
            image encoder, for example. Image encoder typically has a linear layer
            at the end, and we don't want the MLP to immediately stack another linear
            layer on the input features.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_input_norm: see `add_input_activation`, whether to add a normalization layer
            to the input _before_ the MLP computation.
            values: True to add the `norm_type` to the input
        add_output_activation: whether to add a nonlinearity to the output _after_ the
            MLP computation.
            - True to add the same activation as the rest of the MLP
            - str to add an activation of a different type.
        add_output_norm: see `add_output_activation`, whether to add a normalization layer
            _after_ the MLP computation.
            values: True to add the `norm_type` to the input
    """
    assert (hidden_depth is None) != (num_layers is None), (
        "Either hidden_depth or num_layers must be specified, but not both. "
        "num_layers is defined as hidden_depth+1"
    )
    if hidden_depth is not None:
        assert hidden_depth >= 0
    if num_layers is not None:
        assert num_layers >= 1
    hidden_depth = num_layers - 1 if hidden_depth is None else hidden_depth

    act_layer = get_activation(activation)
    weight_init = get_initializer(weight_init, activation)
    bias_init = get_initializer(bias_init, activation)
    norm_type = get_norm_layer(norm_type)

    modules = [nn.Linear(input_dim, output_dim)]

    if hidden_depth != 0:
        modules.extend([norm_type(hidden_dim), act_layer()])
        for _ in range(hidden_depth - 1):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), norm_type(hidden_dim), act_layer()])
        modules.append(nn.Linear(hidden_dim, output_dim))

    if add_input_norm:
        modules = [norm_type(input_dim), *modules]

    if add_input_activation:
        if add_input_activation is not True:
            act_layer = get_activation(add_input_activation)
        modules = [act_layer(), *modules]
    if add_output_norm:
        modules.append(norm_type(output_dim))
    if add_output_activation:
        if add_output_activation is not True:
            act_layer = get_activation(add_output_activation)
        modules.append(act_layer())

    logger.debug("Initialising weights/biases for linears in MLP...")
    for mod in modules:
        if isinstance(mod, nn.Linear):
            weight_init(mod.weight)
            bias_init(mod.bias)

    return nn.Sequential(*modules)
