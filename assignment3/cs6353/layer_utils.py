# layer_utils.py

from cs6353.layers import *
import numpy as np

def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: Tuple of (fc_cache, relu_cache) from affine_relu_forward

    Returns a tuple of:
    - dx: Gradient with respect to input x
    - dw: Gradient with respect to weights w
    - db: Gradient with respect to biases b
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an affine transform, followed by batch normalization,
    and then a ReLU activation.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights and biases for the affine layer
    - gamma, beta: Scale and shift parameters for batch normalization
    - bn_param: Dictionary of parameters for batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: Tuple of (fc_cache, bn_cache, relu_cache) from affine_bn_relu_forward

    Returns a tuple of:
    - dx: Gradient with respect to input x
    - dw: Gradient with respect to weights w
    - db: Gradient with respect to biases b
    - dgamma: Gradient with respect to scale parameter gamma
    - dbeta: Gradient with respect to shift parameter beta
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn_out = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn_out, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    Convenience layer that performs an affine transform, followed by layer normalization,
    and then a ReLU activation.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights and biases for the affine layer
    - gamma, beta: Scale and shift parameters for layer normalization
    - ln_param: Dictionary of parameters for layer normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    ln_out, ln_cache = layernorm_forward(a, gamma, beta, ln_param)
    out, relu_cache = relu_forward(ln_out)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache


def affine_ln_relu_backward(dout, cache):
    """
    Backward pass for the affine-layernorm-relu convenience layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: Tuple of (fc_cache, ln_cache, relu_cache) from affine_ln_relu_forward

    Returns a tuple of:
    - dx: Gradient with respect to input x
    - dw: Gradient with respect to weights w
    - db: Gradient with respect to biases b
    - dgamma: Gradient with respect to scale parameter gamma
    - dbeta: Gradient with respect to shift parameter beta
    """
    fc_cache, ln_cache, relu_cache = cache
    dln_out = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = layernorm_backward(dln_out, ln_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
