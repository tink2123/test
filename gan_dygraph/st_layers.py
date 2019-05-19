from __future__ import division
import paddle.fluid as fluid
import numpy as np
import os

# cudnn is not better when batch size is 1.
use_cudnn = False
if 'ce_mode' in os.environ:
    use_cudnn = False

###def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
###    inv = fluid.layers.sqrt(variance + variance_epsilon)
###    if scale is not None:
###        inv = fluid.layers.elementwise_mul(x=inv, y=scale, axis=1)
###    return x * fluid.layers.cast(inv, x.dtype)) + \
###       fluid.layers.cast(offset - mean* inv if offset is not None else -mean * inv, x.dtype)

def norm_layer(input, norm_type='batch_norm', name=None):
    if norm_type == 'batch_norm':
        param_attr = fluid.ParamAttr(
             name = name+'_w', 
             initializer = fluid.initializer.NormalInitializer(loc=1.0, scale=0.02))
        bias_attr = fluid.ParamAttr(
             name = name+'_b', 
             initializer = fluid.initializer.Constant(value=0.0))
        return fluid.layers.batch_norm(
                     input, is_test=False, param_attr=param_attr, bias_attr=bias_attr,
                     moving_mean_name=name+'_mean', moving_variance_name=name+'_var') 
###                     do_model_average_for_mean_and_var=True)

    elif norm_type == 'instance_norm':
        helper = fluid.layer_helper.LayerHelper("instance_norm", **locals())
        dtype = helper.input_dtype()
        epsilon = 1e-5
        mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        var = fluid.layers.reduce_mean(
            fluid.layers.square(input - mean), dim=[2, 3], keep_dim=True)
        if name is not None:
            scale_name = name + "_scale"
            offset_name = name + "_offset"
        scale_param = fluid.ParamAttr(
            name=scale_name,
###            initializer=fluid.initializer.TruncatedNormal(1.0, 0.02),
            initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=0.02),
            trainable=True)
        offset_param = fluid.ParamAttr(
            name=offset_name,
            initializer=fluid.initializer.Constant(0.0),
            trainable=True)
        scale = helper.create_parameter(
            attr=scale_param, shape=input.shape[1:2], dtype=dtype)
        offset = helper.create_parameter(
            attr=offset_param, shape=input.shape[1:2], dtype=dtype)
###        return batch_normalization(input, mean, var, offset, scale, epsilon, name="instance_norm")
    
        tmp = fluid.layers.elementwise_mul(x=(input - mean), y=scale, axis=1)
        tmp = tmp / fluid.layers.sqrt(var + epsilon)
        tmp = fluid.layers.elementwise_add(tmp, offset, axis=1)
        return tmp
    else:
        raise NotImplementedError("[%s] in not support" % type)


def conv2d(input,
           num_filters=64,
           filter_size=7,
           stride=1,
           stddev=0.02,
           padding=0,
           name="conv2d",
           norm=True,
           relu=True,
           relufactor=0.0,
           use_bias=False):
    """Wrapper for conv2d op to support VALID and SAME padding mode."""
    param_attr = fluid.ParamAttr(
        name=name + "_w",
        initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=stddev))
    if use_bias == True:
        bias_attr = fluid.ParamAttr(
            name=name + "_b", 
            initializer=fluid.initializer.Constant(0.0))
###            initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=stddev))
    else:
       bias_attr = False

    conv = fluid.layers.conv2d(
        input,
        num_filters,
        filter_size,
        name=name,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)
    if norm:
        conv = norm_layer(input=conv, norm_type='batch_norm', name=name + "_norm")
        #print conv
        #fluid.layers.Print(input=conv, print_tensor_name=True,summarize=2)
    if relu:
        conv = fluid.layers.leaky_relu(conv, alpha=relufactor)
    return conv


def deconv2d(input,
             num_filters=64,
             filter_size=7,
             stride=1,
             stddev=0.02,
             padding=[0,0],
             outpadding=[0,0,0,0],
             name="deconv2d",
             norm=True,
             relu=True,
             relufactor=0.0,
             use_bias=False):
    """Wrapper for deconv2d op to support VALID and SAME padding mode."""
    param_attr = fluid.ParamAttr(
        name=name + "_w",
        initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=stddev))
    if use_bias == True:
        bias_attr = fluid.ParamAttr(
            name=name + "_b",
            initializer=fluid.initializer.Constant(0.0))
###            initializer = fluid.initializer.NormalInitializer(loc=0.0, scale=stddev))
    else:
        bias_attr = False

    conv = fluid.layers.conv2d_transpose(
        input,
        num_filters,
        name=name,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        use_cudnn=use_cudnn,
        param_attr=param_attr,
        bias_attr=bias_attr)

    conv = fluid.layers.pad2d(conv, paddings=outpadding, mode='constant', pad_value=0.0)

    if norm:
        conv = norm_layer(input=conv, norm_type='batch_norm', name=name + "_norm")
    if relu:
        conv = fluid.layers.leaky_relu(conv, alpha=relufactor)
    return conv
